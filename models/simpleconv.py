# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from tempfile import tempdir
import typing as tp
import torch
from config import SimpleConvConfig
from torch import nn
from torch.nn import functional as F

from studies.study import Recording
from .common import (
    ConvSequence,
    ConditionalLayers,
    ChannelDropout,
)
from .channel_merger import ChannelMerger
from .quantizer import Quantizer, VQQuantizer, GumbelQuantizer
from .transformer import TransformerEncoder, TransformerDecoder
from losses import CLIPLoss


class SimpleConv(nn.Module):
    def __init__(
        self,
        config: SimpleConvConfig,
    ):
        super().__init__()

        self.config = config
        channels = self.config.in_channels

        assert (
            self.config.kernel_size % 2 == 1
        ), "For padding to work, this must be verified"

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
        if self.config.batch_norm:
            self.initial_batch_norm = nn.BatchNorm1d(channels)

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

        # Convolutional blocks parameter
        # Compute the sequences of channel sizes
        conv_channel_sizes = [channels] + [
            int(round(self.config.hidden_dim * self.config.growth**k))
            for k in range(self.config.depth)
        ]
        params: tp.Dict[str, tp.Any]
        params = dict(
            kernel=self.config.kernel_size,
            stride=1,
            dropout=self.config.conv_dropout,
            dropout_input=self.config.dropout_input,
            batch_norm=self.config.batch_norm,
            dilation_growth=self.config.dilation_growth,
            dilation_period=self.config.dilation_period,
            glu=self.config.glu,
            activation=nn.GELU,
        )
        self.encoders = ConvSequence(conv_channel_sizes, **params)
        final_channels = conv_channel_sizes[-1]

        # Final transformer encoder
        self.transformer_encoders, self.quantizer, self.layer_norm = False, False, False
        if self.config.transformer_encoder_layers > 0:

            self.layer_norm = nn.LayerNorm(normalized_shape=final_channels)

            assert self.config.transformer_input in [
                "continuous",
                "quantized",
                "concat",
            ], f"Invalid transformer input {self.config.transformer_input}"

            # Quantizer
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
                    final_channels *= 2

            self.transformer_encoders = TransformerEncoder(
                d_model=final_channels,
                nhead=self.config.transformer_encoder_heads,
                dropout=self.config.conv_dropout,
                layers=self.config.transformer_encoder_layers,
                embedding=self.config.transformer_encoder_emb,
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
        self.final = None
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
        cnn_params = sum(p.numel() for p in self.encoders.parameters())
        conditions = (
            list(self.config.conditions.keys()) if self.config.conditions else []
        )
        print(
            f"SimpleConv initialized with {total_params} parameters, cond: {conditions}"
        )
        print(
            f"Merger {self.merger is not None}, merger channels {self.config.merger_channels}"
        )
        print(
            f"ConvBlocks: {self.config.depth}, hidden_dim: {self.config.hidden_dim}, params {cnn_params}"
        )

        # Leave the temp param
        self.clip_loss = CLIPLoss()

    def forward(
        self,
        x: torch.Tensor,
        recording: Recording,
        conditions: tp.Dict[str, str] = None,
        mel: torch.Tensor = None,
        train: bool = False,
    ):
        """
        Arguments:
            x -- meg scans of shape [B, C, T]
            recording -- Recording object with the layout and subject index
            conditions -- dictionary of conditions_type : condition_name
            torch.Tensor -- mel spectrogram of shape [B, mel_bins, T], UNSHIFTED.
        """
        length = x.shape[-1]

        # For transformer later, to not attend to padding time steps, of shape [B, T]
        if self.transformer_encoders:
            mask_shape_tensor = x.clone().transpose(1, 2)  # [B, T, C]
            sequence_condition = mask_shape_tensor.sum(dim=2) == 0  # [B, T]
            attention_mask = (
                sequence_condition  # True for padding positions (to be masked)
            )
            attention_mask = attention_mask.to(x.device)
        else:
            attention_mask = None

        if self.dropout is not None:
            x = self.dropout(x=x, recording=recording)

        if self.merger is not None:

            if self.config.merger_conditional is not None:
                assert (
                    self.config.merger_conditional in conditions.keys()
                ), f"The merger conditional type {self.config.merger_conditional} must be in the conditions"
                condition = conditions[self.config.merger_conditional]
            else:
                condition = None

            x = self.merger(
                x=x,
                recording=recording,
                condition=condition,
            )

        if self.initial_batch_norm is not None:
            x = self.initial_batch_norm(x)

        if self.initial_linear is not None:
            x = self.initial_linear(x)

        if self.conditional_layers is not None:
            for cond_type, cond_layer in self.conditional_layers.items():
                assert (
                    cond_type in conditions.keys()
                ), f"The conditional type {cond_type} must be in the conditions"
                x = cond_layer(x, condition=conditions[cond_type])

        # CNN
        x = self.encoders(x)  # [B, C, T]

        # Transformers
        decoder_inference, quantizer_metrics = False, None
        if self.transformer_encoders:

            if self.layer_norm:
                x = self.layer_norm(x.transpose(1, 2)).transpose(  # [B, T, C]
                    1, 2
                )  # [B, C, T]

            if self.quantizer:

                quantized, quantizer_metrics = self.quantizer(x)  # [B, C, T]

                if self.config.transformer_input == "concat":
                    x = torch.cat([x, quantized], dim=1)  # [B, 2C, T]
                elif self.config.transformer_input == "quantized":
                    x = quantized

            x = self.transformer_encoders(x, attn_mask=attention_mask)  # [B, C, T]

            if self.transformer_decoders:
                if train:
                    assert (
                        mel is not None
                    ), "Mel spectrogram must be provided for training"

                    (b_1, _, t_1), (b_2, _, t_2) = x.shape, mel.shape

                    assert (
                        b_1 == b_2 and t_1 == t_2
                    ), f"Batch size and time steps must match. {b_1} != {b_2} or {t_1} != {t_2}"

                    # Shift mel spectrogram to the right
                    mel = F.pad(mel, (1, 0), "constant", 0)[
                        ..., :t_2
                    ]  # [B, mel_bins, T]
                    x = self.transformer_decoders(
                        mel=mel, encoder_output=x, src_mask=attention_mask
                    )  # [B, C, T]

                # Inference with auto-regressive decoding
                else:
                    self.transformer_decoders.eval()
                    with torch.no_grad():

                        mel_bins = self.transformer_decoders.mel_bins
                        B, C, T = x.shape
                        mel = torch.zeros(B, mel_bins, 1).to(x.device)  # [B, mel, 1]

                        for t in range(T):
                            output = self.transformer_decoders(
                                mel=mel, encoder_output=x, src_mask=attention_mask
                            )  # [B, d_model, t + 1]

                            # [B, d_model, 1] -> [B, mel, 1]
                            next_mel = self.final(output[..., -1:])
                            mel = torch.cat([mel, next_mel], dim=-1)  # [B, mel, t + 1]

                        # [B, mel_bins, T + 1] -> [B, mel_bins, T]
                        x = mel[..., 1:]
                        decoder_inference = True

        # Final projection, except when it is alreay done in the decoder
        if not decoder_inference:
            x = self.final(x)  # [B, C, T]

            # Follow Whisper's mel spectrogram normalization
            if self.config.mel_normalization:
                x = F.relu(x)  # Positive energy
                x = torch.clamp(x, min=1e-10).log10()
                # Compresses dynamic range
                # Noise floor relative to loudest part of each frame
                x = torch.maximum(x, x.max(-1, keepdim=True)[0] - 8.0)
                # Normalize to [0, 1], shift negative log values to positive
                x = (x + 4.0) / 4.0

        assert x.shape[-1] == length
        return x, quantizer_metrics
