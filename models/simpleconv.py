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
from torch import nn
from torch.nn import functional as F
from transformer import Transformer

from ..studies.study import Recording

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

        if config.conditions is not None:

            assert (
                len(self.config.conditions) > 0
            ), "There must be at least one condition"

            # double dictionary to map condition type and name to index
            self.condition_to_idx = {}
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
                embedding_dim=self.config.merger_emb_dim,
                dropout=self.config.merger_dropout,
                conditions=conditions,
                layout_dim=self.config.layout_dim,
                layout_proj=self.config.layout_proj,
                layout_scaling=self.config.layout_scaling,
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
            self.transformer_encoders = Transformer(
                d_model=final_channels,
                nhead=self.config.transformer_heads,
                dropout=self.config.conv_dropout,
                layers=self.config.transformer_layers,
                embedding=None,
                causal=self.config.is_causal,
                use_attention_mask=self.config.use_attention_mask,
                concat_spectrals=self.config.transformer_concat_spectrals,
                bins=self.config.transformer_bins,
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
        print(f"\nSimpleConv: \n\tParams: {total_params}")
        print(
            f"\tConv blocks: {self.config.depth}\n\tTrans layers: {self.config.transformer_layers}"
        )
        print(
            f"Spectral: {self.config.transformer_concat_spectrals}, Decoder: {self.config.transformer_decoder}"
        )
        print(
            f"Causal: {self.config.is_causal}, Attention mask: {self.config.use_attention_mask}"
        )

    def forward(
        self,
        x: torch.Tensor,
        recording: Recording,
        conditions: tp.Dict[str, str],
    ):
        """
        Arguments:
            x -- meg scans of shape [B, C, T]
            recording -- Recording object with the layout and subject index
            conditions -- dictionary of conditions_type : condition_name
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
            x = self.dropout(x=x, recording=recording)

        if self.merger is not None:
            assert (
                self.config.merger_conditional in conditions.keys()
            ), f"The merger conditional type {self.config.merger_conditional} must be in the conditions"

            x = self.merger(
                x=x,
                recording=recording,
                condition=conditions[self.config.merger_conditional],
            )

        if self.initial_linear is not None:
            x = self.initial_linear(x)

        for cond_type, cond_layer in self.conditional_layers.items():
            assert (
                cond_type in conditions.keys()
            ), f"The conditional type {cond_type} must be in the conditions"
            x = cond_layer(x, condition=conditions[cond_type])

        # CNN
        x = self.encoders(x)  # [B, C, T]

        # Transformers
        if self.transformer_encoders:
            self.transformer_encoders(x, attn_mask=attention_mask)  # [B, C, T]

        # Final projection
        x = self.final(x)  # [B, C, T]
        assert x.shape[-1] >= length

        return x
