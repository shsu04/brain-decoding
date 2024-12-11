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
    ConditionalLayers,
    ChannelMerger,
    ChannelDropout,
)


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

        # Spatial dropout and rescale
        self.dropout = None
        if self.config.dropout > 0.0:
            self.dropout = ChannelDropout(self.config.dropout)

        # Channel merger by spatial attention
        self.merger = None
        if self.config.merger:
            self.merger = ChannelMerger(
                merger_channels=self.config.merger_channels,
                embedding_dim=self.config.merger_emb_dim,
                dropout=self.config.merger_dropout,
                n_conditions=self.config.merger_conditions,
                conditional=self.config.merger_conditional,
            )
            channels = self.config.merger_channels

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

        # Subject-specific layers
        self.conditional_layers = None
        if self.config.conditional_layers:
            dim = {
                "self.config.hidden_dim": self.config.hidden_dim,
                "input": channels,
            }[self.config.conditional_layers_dim]
            self.conditional_layers = ConditionalLayers(
                in_channels=channels,
                out_channels=dim,
                n_conditions=self.config.n_conditions,
            )
            channels = dim

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
        self.final = nn.Sequential(
            nn.Conv1d(final_channels, 2 * final_channels, 1),
            nn.GELU(),
            nn.ConvTranspose1d(
                2 * final_channels, self.config.out_channels, kernel, stride, pad
            ),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"\nSimpleConv: \n\tParams: {total_params}\n\tConv blocks: {self.config.depth}\n\tTrans layers: {self.config.transformer_layers}"
        )
        print(f"Convolutional channel sizes: {conv_channel_sizes}")

    def forward(
        self,
        x: torch.Tensor,
        layout: torch.Tensor = None,
        subjects: torch.Tensor = None,
    ):
        """
        Arguments:
            x -- meg scans of shape [B, C, T]

        Keyword Arguments:
            layout -- layout tensor of shape [C, 2] with the channel positions
            subjects -- tensor of shape [B] with the subject index for each sample
        """
        length = x.shape[-1]

        # For transformer later, to not attend to padding time steps, of shape [B, T]
        if self.transformer_encoders and self.config.use_attention_mask:
            mask_shape_tensor = x.clone().permute(0, 2, 1)
            sequence_condition = mask_shape_tensor.sum(dim=2) == 0  # [B, T]

            attention_mask = torch.zeros_like(sequence_condition).float()
            attention_mask[sequence_condition] = float("-inf")  # mask padding

            attention_mask = attention_mask.to(x.device)
        else:
            attention_mask = None

        if self.dropout is not None:
            x = self.dropout(x=x, layout=layout)
        if self.merger is not None:
            x = self.merger(x=x, layout=layout)
        if self.initial_linear is not None:
            x = self.initial_linear(x)
        if self.conditional_layers is not None:
            x = self.conditional_layers(x, subjects)

        # CNN
        x = self.encoders(x)

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
        x = self.final(x)

        assert x.shape[-1] >= length
        x = x[:, :, :length]  # [B, C, T]

        return x
