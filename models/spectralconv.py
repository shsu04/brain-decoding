import torch
import typing as tp
from studies.study import Recording
from torch import nn
from torch.nn import functional as F
import torchaudio.transforms as T

from studies.study import Recording
from .common import (
    ConditionalLayers,
    ChannelDropout,
)
from .channel_merger import ChannelMerger
from .spectral_conv_sequence import SpectralConvSequence
from .quantizer import VQQuantizer, GumbelQuantizer
from .rnn import RNNEncoder, TransformerDecoder
from losses import CLIPLoss
from config import SpectralConvConfig


class SpectralConv(torch.nn.Module):
    """
    Variant of SimpleConv that handles spectrogram representation of the data.
    Except for ..., all other steps (before group norm and after the last conv layer)
    remain identical to SimpleConv.
    """

    def __init__(self, config: SpectralConvConfig):
        super().__init__()
        self.config = config
        channels = self.config.in_channels

        # Condition dictionary generator
        self.condition_to_idx = {}
        if config.conditions is not None:

            assert (
                len(self.config.conditions) > 0
            ), "There must be at least one condition"

            # double dictionary to map condition type and name to index
            for cond_type, cond_names in sorted(self.config.conditions.items()):
                # add an additional condition in case not found during training
                self.condition_to_idx[cond_type] = {
                    cond: idx
                    for idx, cond in enumerate(sorted(cond_names + ["unknown"]))
                }

        # Spatial dropout and rescale
        self.dropout = None
        if self.config.dropout > 0.0:
            self.dropout = ChannelDropout(
                self.config.dropout,
                layout_dim=self.config.layout_dim,
                layout_proj=self.config.layout_proj,
                layout_scaling=self.config.layout_scaling,
            )

        # Channel merger by spatial attention
        self.merger = None
        if self.config.merger:

            if self.config.merger_conditional is not None:
                assert (
                    config.merger_conditional in self.condition_to_idx.keys()
                ), f"The merger conditional type {config.merger_conditional} must be in the conditions"
                conditions = self.condition_to_idx[config.merger_conditional]
            else:
                conditions = None

            self.merger = ChannelMerger(
                merger_channels=self.config.merger_channels,
                embedding_type=self.config.merger_emb_type,
                embedding_dim=self.config.merger_emb_dim,
                dropout=self.config.merger_dropout,
                conditions=conditions,
                layout_dim=self.config.layout_dim,
                layout_proj=self.config.layout_proj,
                layout_scaling=self.config.layout_scaling,
            )
            channels = self.config.merger_channels

        # Batch norm if needed, each channel independently
        self.initial_batch_norm = None
        if self.config.initial_batch_norm:
            self.initial_batch_norm = nn.BatchNorm2d(num_features=channels)

        # SAME AS SIMPLE CONV, FLATTENED SPECTROGRAM
        # Project MEG channels with a linear layer
        self.initial_linear = None
        if self.config.initial_linear:
            init = [nn.Conv1d(channels, self.config.initial_linear, 1)]
            for _ in range(self.config.initial_depth - 1):
                init += [
                    nn.GELU(),
                    nn.Conv1d(
                        self.config.initial_linear, self.config.initial_linear, 1
                    ),
                ]
            self.initial_linear = nn.Sequential(*init)
            channels = self.config.initial_linear

        # SAME AS SIMPLE CONV, FLATTENED SPECTROGRAM
        # Conditional layers
        self.conditional_layers = nn.ModuleDict()

        if self.config.conditional_layers:
            for cond_type, cond_names in sorted(self.config.conditions.items()):
                dim = {
                    "hidden_dim": self.config.hidden_dim,
                    "input": channels,
                }[self.config.conditional_layers_dim]

                self.conditional_layers[cond_type] = ConditionalLayers(
                    in_channels=channels,
                    out_channels=dim,
                    conditions=self.condition_to_idx[cond_type],
                )
                channels = dim

                print(
                    f"Conditional layer {cond_type} initialized with {len(self.conditional_layers[cond_type].conditions)} conditions"
                )

        # Channels for conv has to be specified directly unlike in SimpleConv
        assert (
            self.config.cnn_channels[0] == channels
        ), f"First channel must be {channels}, got {self.config.cnn_channels[0]}"

        # Spectrogram CNN encoder
        self.encoders = SpectralConvSequence(
            channels=self.config.cnn_channels,
            kernel=self.config.kernel_size,
            dilation_growth=self.config.dilation_growth,
            dilation_period=self.config.dilation_period,
            stride=1,
            glu=self.config.glu,
            dropout=self.config.conv_dropout,
            dropout_input=self.config.dropout_input,
            batch_norm=self.config.batch_norm,
            activation=nn.GELU,
            half=self.config.half,
            pos_encoding=self.config.cnn_pos_encoding,
            bins=self.config.bins,
        )
        final_channels = self.config.bins * self.config.cnn_channels[-1]

        # Final transformer encoder
        self.rnn_encoders, self.quantizer, self.layer_norm = False, False, False
        if self.config.transformer_encoder_layers > 0:

            assert self.config.transformer_input in [
                "continuous",
                "quantized",
                "concat",
            ], f"Invalid transformer input {self.config.transformer_input}"

            # Quantizer
            self.quantizer_compression = None
            if self.config.quantizer:

                assert self.config.quantizer in [
                    "vq",
                    "gumbel",
                ], f"Invalid quantizer {self.config.quantizer}"

                if self.config.quantizer == "vq":
                    self.quantizer = VQQuantizer(
                        dim=final_channels,
                        num_codebooks=self.config.num_codebooks,
                        codebook_size=self.config.codebook_size,
                        commitment=self.config.quantizer_commitment,
                    )
                else:
                    self.quantizer = GumbelQuantizer(
                        dim=final_channels,
                        num_codebooks=self.config.num_codebooks,
                        codebook_size=self.config.codebook_size,
                        temp_init=self.config.quantizer_temp_init,
                        temp_min=self.config.quantizer_temp_min,
                        temp_decay=self.config.quantizer_temp_decay,
                    )

                if self.config.transformer_input == "concat":
                    self.quantizer_compression = nn.Conv1d(
                        final_channels * 2, final_channels, 1
                    )

            self.rnn_encoders = RNNEncoder(
                d_model=final_channels,
                nhead=self.config.transformer_encoder_heads,
                dropout=self.config.conv_dropout,
                layers=self.config.transformer_encoder_layers,
                embedding=self.config.transformer_encoder_emb,
                rnn_type=self.config.rnn_type,
                # Conformer params
                depthwise_conv_kernel_size=self.config.depthwise_conv_kernel_size,
                use_group_norm=self.config.use_group_norm,
                convolution_first=self.config.convolution_first,
            )

        # Final transformer decoder
        self.transformer_decoders = False
        if self.config.transformer_decoder_layers > 0:
            assert (
                self.config.transformer_encoder_layers > 0
            ), "Must have transformer encoders to use decoders"
            self.transformer_decoders = TransformerDecoder(
                encoder_output_dim=final_channels,
                d_model=self.config.transformer_decoder_dim,
                nhead=self.config.transformer_decoder_heads,
                dropout=self.config.conv_dropout,
                layers=self.config.transformer_decoder_layers,
                embedding=self.config.transformer_decoder_emb,
            )
            final_channels = self.config.transformer_decoder_dim

        # Final encoder linear projection
        pad, kernel, stride = 0, 1, 1
        self.final = nn.Sequential(
            nn.Conv1d(final_channels, 2 * final_channels, 1),
            nn.GELU(),
            nn.ConvTranspose1d(
                2 * final_channels, self.config.out_channels, kernel, stride, pad
            ),
        )
        nn.init.kaiming_uniform_(self.final[0].weight, a=0)

        total_params = sum(p.numel() for p in self.parameters())
        conditional_params = sum(
            p.numel() for p in self.conditional_layers.parameters()
        )
        cnn_params = sum(p.numel() for p in self.encoders.parameters())
        conditions = (
            list(self.config.conditions.keys()) if self.config.conditions else []
        )
        print(
            f"SpectralConv initialized with {total_params - conditional_params} parameters, cond: {conditions}"
        )
        print(
            f"Merger {self.merger is not None}, merger channels {self.config.merger_channels}"
        )
        print(
            f"ConvBlocks: {self.config.cnn_channels}, hidden_dim: {self.config.hidden_dim}, params {cnn_params}"
        )

        self.spectrogram_transform = T.Spectrogram(
            n_fft=self.config.bins * 2,
            hop_length=self.config.hop_length,
            normalized=True,
            power=2,
        )

        # Leave the temp param
        self.clip_loss = CLIPLoss(dim=self.config.out_channels)

    def forward(
        self,
        x: tp.List[torch.Tensor],
        recording: tp.List[Recording],
        conditions: tp.List[tp.Dict[str, str]] = None,
        mel: tp.List[torch.Tensor] = None,
        train: bool = False,
        return_hidden_outputs: bool = False,
    ):
        """
        Arguments:
            x -- list of meg scans of shape [B, C, T]
            recording -- list of Recording object with the layout and subject index
            conditions -- list of dictionary of conditions_type : condition_name
            mel -- list of mel spectrogram of shape [B, mel_bins, T], UNSHIFTED.
            train -- boolean flag to indicate training or inference
            return_hidden_outputs -- flag to return hidden outputs from CNN and RNNs, [B, C, mel, T] of length L

        Returns:
            x -- output of the model, [B, C, T]
            quantizer_metrics -- metrics from the quantizer, if used.
            channel_weights -- list of batch size * [B, C, C']
            hidden_outputs -- list of hidden outputs from CNN and RNNs, [B, C, T] of length L
        """
        x_aggregated, channel_weights = [], []
        condition_indices_map = {cond_type: [] for cond_type in self.condition_to_idx}
        attention_mask = None
        self.spectrogram_transform = self.spectrogram_transform.to(x[0].device)

        # Merge and gather condition indices per batch
        for i in range(len(x)):

            x_i, recording_i, conditions_i = x[i], recording[i], conditions[i]

            # Convert to spectrogram [B, C, T] -> [B, C, mel, T]
            B, C, M, T = (
                x_i.size(0),
                x_i.size(1),
                self.config.bins,
                x_i.size(2) // self.config.hop_length,
            )
            x_i = torch.log1p(self.spectrogram_transform(x_i))
            x_i = x_i[..., :M, :T]  #  ensure correct size [B, C, mel, T]

            # [B, C, mel, T] -> [B, C, mel * T]
            x_i = x_i.reshape(B, C, M * T)

            if self.dropout is not None:
                x_i = self.dropout(x=x_i, recording=recording_i)

            # Merger, all batches onwards should have same number of channels
            if self.merger is not None:
                if self.config.merger_conditional is not None:
                    cond_name = conditions_i.get(
                        self.config.merger_conditional, "unknown"
                    )
                else:
                    cond_name = None
                x_i, channel_weight = self.merger(
                    x=x_i, recording=recording_i, condition=cond_name
                )
                channel_weights.append(channel_weight)

            # Gather condition indices, one per condition type
            for cond_type in self.condition_to_idx:
                c_str = conditions_i.get(cond_type, "unknown")  # name
                idx = self.condition_to_idx[cond_type].get(
                    c_str, self.condition_to_idx[cond_type]["unknown"]
                )  # index
                index_tensor = torch.full(
                    (x_i.size(0),), idx, dtype=torch.long, device=x_i.device
                )  # [B_i]
                condition_indices_map[cond_type].append(index_tensor)

            x_aggregated.append(x_i)

        # CONCATENATE BATCHES
        x = torch.cat(x_aggregated, dim=0)  # [B_i * i, C, mel * T]
        del x_aggregated
        B = x.size(0)

        x = x.reshape(B, x.size(1), M, T)  # [B, C, mel * T] -> [B, C, mel, T]

        condition_indices_map = {
            cond_type: torch.cat(indices, dim=0)
            for cond_type, indices in condition_indices_map.items()
        }  # Each cond type has a tensor of indices [B_i * i] = [B]

        if mel is not None and len(mel) > 0:
            mel = torch.cat(mel, dim=0)  # [B_i * i, mel_bins, T]

        if self.initial_batch_norm is not None:
            x = self.initial_batch_norm(x)

        # [B, C, mel, T] -> [B, C, mel * T] to re-use old code since only acts on channels
        x = x.reshape(B, x.size(1), M * T)

        if self.initial_linear is not None:
            x = self.initial_linear(x)

        if self.conditional_layers is not None:
            for cond_type, cond_layer in self.conditional_layers.items():
                x = cond_layer(x, condition_indices_map[cond_type])

        # [B, C, mel * T] -> [B, C, mel, T] for spectrogram CNN encoder
        x = x.reshape(B, x.size(1), M, T)

        # CNN
        x, hidden_outputs = self.encoders(x)  # [B, C, mel, T]
        x = x.reshape(B, x.size(1) * M, T)  # [B, C, mel, T] -> [B, C * mel, T]

        # Transformers
        decoder_inference, quantizer_metrics = False, None
        if self.rnn_encoders:

            if self.quantizer:
                quantized, quantizer_metrics = self.quantizer(x)  # [B, C, T]
                if self.config.transformer_input == "concat":
                    x = torch.cat([x, quantized], dim=1)  # [B, 2C, T]
                    x = self.quantizer_compression(x)  # [B, C, T]
                elif self.config.transformer_input == "quantized":
                    x = quantized

            # Leave this for now until recordings come with attn mask
            attention_mask = None
            x, rnn_hidden_outputs = self.rnn_encoders(
                x, attn_mask=attention_mask, return_hidden_outputs=return_hidden_outputs
            )  # [B, C, T]

            # Save intermediate hidden outputs, del for memory
            if return_hidden_outputs:
                hidden_outputs.extend(rnn_hidden_outputs)
            del rnn_hidden_outputs

            if self.transformer_decoders:
                if train:
                    assert (
                        mel is not None
                    ), "Mel spectrogram must be provided for decoder training"

                    (b_1, c_1, t_1), (b_2, mel_bins, t_2) = x.shape, mel.shape
                    assert b_1 == b_2 and t_1 == t_2, (
                        f"Encoder shape [B={b_1},C={c_1},T={t_1}] does not match "
                        f"decoder shape [B={b_2},mel={mel_bins},T={t_2}]"
                    )

                    # Shift mel spectrogram to the right
                    mel = F.pad(mel, (1, 0), "constant", 0)[
                        ..., :t_2
                    ]  # [B, mel_bins, T]
                    x, decoder_hidden_outputs = self.transformer_decoders(
                        mel=mel,
                        encoder_output=x,
                        src_mask=attention_mask,
                        return_hidden_outputs=return_hidden_outputs,
                    )  # [B, C, T]

                    # Save intermediate hidden outputs, del for memory
                    if return_hidden_outputs:
                        hidden_outputs.extend(decoder_hidden_outputs)
                    del decoder_hidden_outputs

                else:
                    # Inference with auto-regressive decoding
                    self.transformer_decoders.eval()

                    with torch.no_grad():

                        mel_bins = self.transformer_decoders.mel_bins
                        B, C, T = x.shape
                        mel = torch.zeros(B, mel_bins, 1).to(x.device)  # [B, mel, 1]

                        for t in range(T):
                            output, decoder_hidden_outputs = self.transformer_decoders(
                                mel=mel,
                                encoder_output=x,
                                src_mask=attention_mask,
                                return_hidden_outputs=return_hidden_outputs,
                            )  # [B, d_model, t + 1]

                            # Save intermediate hidden outputs, del for memory
                            if t == T - 1 and return_hidden_outputs:
                                hidden_outputs.extend(decoder_hidden_outputs)
                            del decoder_hidden_outputs

                            output = self.final(output)  # [B, C, t + 1]

                            if self.config.mel_normalization:
                                output = self.whisper_normalization(output)

                            # [B, d_model, 1] -> [B, mel, 1]
                            next_mel = output[..., -1:]
                            mel = torch.cat([mel, next_mel], dim=-1)  # [B, mel, t + 1]

                        # [B, mel_bins, T + 1] -> [B, mel_bins, T]
                        x = mel[..., 1:]

                    decoder_inference = True

        # Final projection (unless the decoder already produced x)
        if not decoder_inference:
            x = self.final(x)  # [B, C, T]

            if self.config.mel_normalization:
                x = self.whisper_normalization(x)

        return x, quantizer_metrics, channel_weights, hidden_outputs

    def whisper_normalization(self, x: torch.Tensor):
        """Follow Whisper's mel spectrogram normalization"""

        # # Ensure no negative energy
        # x = F.relu(x)
        # x = torch.clamp(x, min=1e-10).log10()

        # # Dynamic range compression
        # batch_max = (
        #     x.view(x.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        # )  # Shape: [B, 1, 1]
        # x = torch.maximum(x, batch_max - 8.0)

        # # Normalize to [0, 1]
        # x = (x + 4.0) / 4.0

        x = 1.25 * torch.tanh(x) + 0.75

        # max_values_batch = torch.max(
        #     x.flatten(start_dim=1, end_dim=2), dim=1
        # ).values  # [B, C, T] -> [B, C * T] -> [B]

        # # [B, 1, 1]
        # max_values_batch = max_values_batch.unsqueeze(-1).unsqueeze(-1)
        # # [B, C, T]
        # x = torch.clip(x, min=max_values_batch - 3, max=max_values_batch)

        return x
