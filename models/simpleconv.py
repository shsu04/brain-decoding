# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import random
import typing as tp
import torch
from config import SimpleConvConfig

# import torch_audio as ta

# import torchaudio as ta
from torch import nn
from torch.nn import functional as F

from .common import (
    ConvSequence,
    ScaledEmbedding,
    SubjectLayers,
    DualPathRNN,
    ChannelMerger,
    ChannelDropout,
    pad_multiple,
)


class SimpleConv(nn.Module):
    def __init__(
        self,
        config: SimpleConvConfig,
    ):
        super().__init__()

        self.config = config

        if set(self.config.in_channels.keys()) != set(self.config.hidden.keys()):
            raise ValueError(
                "Channels and hidden keys must match "
                f"({set(self.config.in_channels.keys())} and {set(self.config.hidden.keys())})"
            )
        self._concatenate = self.config.concatenate
        self.out_channels = self.config.out_channels

        # Activation function
        if self.config.gelu:
            activation = nn.GELU
        elif self.config.relu_leakiness:
            activation = partial(nn.LeakyReLU, self.config.relu_leakiness)
        else:
            activation = nn.ReLU

        assert (
            self.config.kernel_size % 2 == 1
        ), "For padding to work, this must be verified"

        # Subsampling MEG channels to a given number
        self.subsampled_meg_channels: tp.Optional[list] = None
        if self.config.subsample_meg_channels:
            assert "meg" in self.config.in_channels
            indexes = list(range(self.config.in_channels["meg"]))
            rng = random.Random(1234)
            rng.shuffle(indexes)
            self.subsampled_meg_channels = indexes[: self.config.subsample_meg_channels]

        # Spatial dropout and rescale
        self.dropout = None
        if self.config.dropout > 0.0:
            self.dropout = ChannelDropout(
                self.config.dropout, self.config.dropout_rescale
            )

        # Channel merger by spatial attention
        self.merger = None
        if self.config.merger:
            self.merger = ChannelMerger(
                merger_channels=self.config.merger_channels,
                embedding_dim=self.config.merger_pos_dim,
                dropout=self.config.merger_dropout,
                usage_penalty=self.config.merger_penalty,
                n_subjects=self.config.n_subjects,
                per_subject=self.config.merger_per_subject,
            )
            self.config.in_channels["meg"] = self.config.merger_channels

        # Project MEG channels with a linear layer
        self.initial_linear = None
        if self.config.initial_linear:
            init = [
                nn.Conv1d(self.config.in_channels["meg"], self.config.initial_linear, 1)
            ]
            for _ in range(self.config.initial_depth - 1):
                init += [
                    activation(),
                    nn.Conv1d(
                        self.config.initial_linear, self.config.initial_linear, 1
                    ),
                ]
            if self.config.initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            self.config.in_channels["meg"] = self.config.initial_linear

        # Subject-specific layers
        self.subject_layers = None
        if self.config.subject_layers:
            assert "meg" in self.config.in_channels
            meg_dim = self.config.in_channels["meg"]
            dim = {"self.config.hidden": self.config.hidden["meg"], "input": meg_dim}[
                self.config.subject_layers_dim
            ]
            self.subject_layers = SubjectLayers(
                meg_dim, dim, self.config.n_subjects, self.config.subject_layers_id
            )
            self.config.in_channels["meg"] = dim

        # STFT, output = [B, C * freq_bins, time_frames]
        self.stft = None
        # if self.config.n_fft is not None:
        #     assert "meg" in self.config.in_channels
        #     self.fft_complex = self.config.fft_complex
        #     self.n_fft = self.config.n_fft
        #     self.stft = ta.transforms.Spectrogram(
        #         n_fft=self.config.n_fft,
        #         hop_length=self.config.n_fft // 2,
        #         normalized=True,
        #         power=None if self.config.fft_complex else 1,
        #         return_complex=True,
        #     )
        #     self.config.in_channels["meg"] *= self.config.n_fft // 2 + 1
        #     if self.config.fft_complex:
        #         self.config.in_channels["meg"] *= 2

        # Learned subject embeddings concatenated to the input
        self.subject_embedding = None
        if self.config.subject_dim:
            self.subject_embedding = ScaledEmbedding(
                num_embeddings=self.config.n_subjects,
                embedding_dim=self.config.subject_dim,
                scale=self.config.embedding_scale,
            )
            self.config.in_channels["meg"] += self.config.subject_dim

        # concatenate inputs if need be
        if self.config.concatenate:
            self.config.in_channels = {"concat": sum(self.config.in_channels.values())}
            self.config.hidden = {"concat": sum(self.config.hidden.values())}

        # Compute the sequences of channel sizes
        sizes = {}
        for name in self.config.in_channels:
            sizes[name] = [self.config.in_channels[name]]
            sizes[name] += [
                int(round(self.config.hidden[name] * self.config.growth**k))
                for k in range(self.config.depth)
            ]

        # Convolutional blocks parameter
        params: tp.Dict[str, tp.Any]
        params = dict(
            kernel=self.config.kernel_size,
            stride=1,
            leakiness=self.config.relu_leakiness,
            dropout=self.config.conv_dropout,
            dropout_input=self.config.dropout_input,
            batch_norm=self.config.batch_norm,
            dilation_growth=self.config.dilation_growth,
            groups=self.config.groups,
            dilation_period=self.config.dilation_period,
            skip=self.config.skip,
            post_skip=self.config.post_skip,
            scale=self.config.scale,
            rewrite=self.config.rewrite,
            glu=self.config.glu,
            glu_context=self.config.glu_context,
            glu_glu=self.config.glu_glu,
            activation=activation,
        )

        final_channels = sum([x[-1] for x in sizes.values()])

        # Dual path RNN
        self.dual_path = None
        if self.config.dual_path:
            self.dual_path = DualPathRNN(final_channels, self.config.dual_path)

        # Comes before the final projection. Transformers
        self.transformer_encoders = False
        if self.config.transformer_layers > 0:
            self.transformer_encoders = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=final_channels,
                    nhead=self.config.transformer_heads,
                    batch_first=True,
                    dropout=self.config.conv_dropout,
                ),
                num_layers=self.config.transformer_layers,
            )

        # Final linear projection
        self.final = None
        pad, kernel, stride = 0, 1, 1

        if self.config.n_fft is not None:
            pad = self.config.n_fft // 4
            kernel = self.config.n_fft
            stride = self.config.n_fft // 2

        if self.config.linear_out:
            assert not self.config.complex_out
            self.final = nn.ConvTranspose1d(
                final_channels, self.config.out_channels, kernel, stride, pad
            )
        elif self.config.complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(
                    2 * final_channels, self.config.out_channels, kernel, stride, pad
                ),
            )
        else:
            assert len(sizes) == 1, "if no linear_out, there must be a single branch."
            params["activation_on_last"] = False
            list(sizes.values())[0][-1] = self.config.out_channels

        # Comes before the final projection. Layers of convolutions.
        self.encoders = nn.ModuleDict(
            {name: ConvSequence(channels, **params) for name, channels in sizes.items()}
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"\nSimpleConv: \n\tParams: {total_params}\n\tConv blocks: {self.config.depth}\n\tTrans layers: {self.config.transformer_layers}"
        )

    def forward(
        self,
        inputs: dict,
        layout: torch.Tensor = None,
        subjects: torch.Tensor = None,
    ):
        """
        Arguments:
            inputs -- dict of {'meg': [B, C, T]}

        Keyword Arguments:
            layout -- layout tensor of shape [C, 2] with the channel positions
            subjects -- tensor of shape [B] with the subject index for each sample
        """
        length = next(iter(inputs.values())).shape[-1]  # length of any of the inputs

        # For transformer later, to not attend to padding time steps, of shape [B, T]
        if self.transformer_encoders and self.config.use_attention_mask:
            mask_shape_tensor = inputs["meg"].clone().permute(0, 2, 1)
            sequence_condition = mask_shape_tensor.sum(dim=2) == 0  # [B, T]

            attention_mask = torch.zeros_like(sequence_condition).float()
            attention_mask[sequence_condition] = float("-inf")  # mask padding

            attention_mask = attention_mask.to(inputs["meg"].device)
        else:
            attention_mask = None

        if self.subsampled_meg_channels is not None:
            mask = torch.zeros_like(inputs["meg"][:1, :, :1])  # Mask [1, C, 1]
            mask[:, self.subsampled_meg_channels] = 1.0
            inputs["meg"] = inputs["meg"] * mask

        if self.dropout is not None:
            inputs["meg"] = self.dropout(x=inputs["meg"], layout=layout)

        if self.merger is not None:
            inputs["meg"] = self.merger(x=inputs["meg"], layout=layout)

        if self.initial_linear is not None:
            inputs["meg"] = self.initial_linear(inputs["meg"])

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)

        # if self.stft is not None:
        #     x = inputs["meg"]
        #     pad = self.n_fft // 4
        #     x = F.pad(pad_multiple(x, self.n_fft // 2), (pad, pad), mode="reflect")
        #     z = self.stft(inputs["meg"])
        #     B, C, Fr, T = z.shape
        #     if self.fft_complex:
        #         z = torch.view_as_real(z).permute(0, 1, 2, 4, 3)
        #     z = z.reshape(B, -1, T)
        #     inputs["meg"] = z

        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            inputs["meg"] = torch.cat(
                [inputs["meg"], emb.expand(-1, -1, length)], dim=1
            )

        if self._concatenate:
            input_list = [x[1] for x in sorted(inputs.items())]
            inputs = {"concat": torch.cat(input_list, dim=1)}

        # CNN
        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](x)

        inputs = [x[1] for x in sorted(encoded.items())]
        x = torch.cat(inputs, dim=1)

        # Dual path RNN
        if self.dual_path is not None:
            x = self.dual_path(x)

        # Transformers
        if self.transformer_encoders:

            x = x.permute(0, 2, 1)  # [B, T, C]

            if self.config.is_causal:
                _, t, _ = x.shape
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    sz=t
                )  # of shape [t, t]
            else:
                causal_mask = None

            x = self.transformer_encoders(
                x,
                mask=causal_mask,
                src_key_padding_mask=attention_mask,
                is_causal=True if (self.config.is_causal) else False,
            )

            x = x.permute(0, 2, 1)  # [B, C, T]

        # Final projection
        if self.final is not None:
            x = self.final(x)

        assert x.shape[-1] >= length
        x = x[:, :, :length]  # [B, C, T]

        return x
