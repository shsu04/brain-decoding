# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import math
import typing as tp
import torch
from torch import nn
import pandas as pd


def pad_multiple(x: torch.Tensor, base: int):
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return torch.nn.functional.pad(x, (0, target - length))


class FourierEmbedding(nn.Module):

    def __init__(self, dimension: int = 2048, margin: float = 0.2):
        super().__init__()

        # Grid size. e.g. dim=2048, n_freqs=32
        n_freqs = (dimension // 2) ** 0.5
        assert int(n_freqs**2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Fourier positional embedding. Maps each channel to a high dimensional space
        through a learnt channel-specific function over sensor locations.
        Unlike trad. embedding this is not using exponential periods for cos and sin,
        but typical `2 pi k` which can represent any function over [0, 1]. As this
        function would be necessarily periodic, we take a bit of margin and do over
        [-0.2, 1.2].

        Arguments:
            positions -- 2D positions of channels in the batch [B, C, 2]

        Returns:
            emb -- Fourier positional embedding [B, C, dim]
        """

        *O, D = positions.shape  # [B, C, 2]
        assert D == 2

        n_freqs = (self.dimension // 2) ** 0.5
        freqs_y = torch.arange(n_freqs).to(positions)  # [n_freqs]
        freqs_x = freqs_y[:, None]  # [n_freqs, 1] for broadcasting
        width = 1 + 2 * self.margin
        positions = positions + self.margin  # Shift pos by margin

        # Scale freq by 2 pi / width to get phase multipliers
        p_x = 2 * math.pi * freqs_x / width  # [n_freqs, 1]
        p_y = 2 * math.pi * freqs_y / width  # [n_freqs]

        # Add dim for broadcasting, [*O, 2] -> [*O, 1, 1, 2]
        positions = positions[..., None, None, :]

        # Compute phase values, (*O, n_freqs, n_freqs) -> (*O, n_freqs*n_freqs)
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)

        # Apply sin and cos to phases and concatenate
        emb = torch.cat(
            [
                torch.cos(loc),
                torch.sin(loc),
            ],
            dim=-1,
        )  # [B, C, dim]

        return emb


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        pass

    def get_positions(self, x: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        """Returns scaled 2D positions of channels in x.

        Arguments:
            x -- recordings with shape [B, C, T]
            layout -- layout of the channels with shape [C, 2]

        Returns:
            torch.Tensor -- 2D positions of channels in x [B, C, 2]
        """
        B, C, T = x.shape
        positions = torch.full((B, C, 2), self.INVALID, device=x.device)

        for idx in range(len(x)):
            positions[idx, : len(layout)] = layout.to(x.device)

        return positions

    def is_invalid(self, positions: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask of invalid channels.

        Arguments:
            positions -- 2D positions of channels in x [B, C, 2]

        Returns:
            torch.Tensor -- boolean mask of invalid channels [B, C]
        """

        return (positions == self.INVALID).all(dim=-1)


class ChannelMerger(nn.Module):
    def __init__(
        self,
        merger_channels: int,
        embedding_dim: int = 2048,
        dropout: float = 0,
        n_conditions: int = 200,
        conditional: bool = False,
    ):
        super().__init__()
        assert embedding_dim % 4 == 0

        self.position_getter = PositionGetter()

        # Learnable heads for each condition
        self.conditional = conditional
        if self.conditional:
            self.heads = nn.Parameter(
                torch.randn(
                    n_conditions, merger_channels, embedding_dim, requires_grad=True
                )
            )
        else:
            self.heads = nn.Parameter(
                torch.randn(merger_channels, embedding_dim, requires_grad=True)
            )

        self.heads.data /= embedding_dim**0.5
        self.dropout = dropout
        self.embedding = FourierEmbedding(dimension=embedding_dim)

    def forward(
        self, x: torch.Tensor, layout: torch.Tensor, conditions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Merges channels by learning a weighted sum of input channels, where the
        weights are determined by the alignment of the channel's positial embeddings
        with learnable heads.

        Arguments:
            x -- input tensor with shape [B, C, T]
            layout -- layout of the channels with shape [C, 2]
            conditions -- tensor of shape [B] with the condition index for each sample

        Returns:
            torch.Tensor -- output tensor with shape [B, merger_channels, T]
        """

        B, C, T = x.shape

        # Spatial embedding, [B, C, 2] -> [B, C, dim]
        positions = self.position_getter.get_positions(x, layout)
        embedding = self.embedding(positions)  # [B, C, dim]

        # Mask out invalid channels, [B, C]
        score_offset = torch.zeros(B, C, device=x.device)
        score_offset[self.position_getter.is_invalid(positions)] = float("-inf")

        # Dropout by random center and radius
        if self.training and self.dropout:

            center_to_ban = torch.rand(2, device=x.device)
            radius_to_ban = self.dropout

            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float("-inf")

        # If conditional, choose heads based on condition
        if self.conditional:
            _, chout, pos_dim = self.heads.shape
            heads = self.heads.gather(
                0, conditions.view(-1, 1, 1).expand(-1, chout, pos_dim)
            )
        else:
            # [ch_out, dim] -> [B, ch_out, dim]
            heads = self.heads[None].expand(B, -1, -1)

        # How well pos emb aligns with learnable heads
        scores = torch.einsum("bcd,bod->boc", embedding, heads)  # [B, C, ch_out]
        scores += score_offset[:, None]  # mask

        # Create each output channel as a weighted sum of input channels
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", x, weights)  # [B, ch_out, T]

        return out


class ConditionalLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conditions: int,
    ):
        super().__init__()

        self.weights = nn.Parameter(
            torch.randn(n_conditions, in_channels, out_channels)
        )
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Applies a conditional linear transformation to the input tensor.

        Arguments:
            x -- input tensor with shape [B, C, T]
            conditions -- tensor with shape [B]

        Returns:
            torch.Tensor -- output tensor with shape [B, C, T]
        """
        _, C, D = self.weights.shape

        # Gather weight matrices for each condition
        weights = self.weights.gather(0, conditions.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"Condition layers({C}, {D}, {S})"


class ChannelDropout(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.position_getter = PositionGetter()

    def forward(self, x: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        """Spatial dropout by random center and radius in normalized [0, 1] coordinates.

        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
        """

        if not self.dropout:
            return x

        B, C, T = x.shape

        # Mask out invalid channels
        positions = self.position_getter.get_positions(x, layout)
        valid = (~self.position_getter.is_invalid(positions)).float()

        x = x * valid[:, :, None]

        if self.training:

            # Spatial dropout by random center and radius
            center_to_ban = torch.rand(2, device=x.device)
            kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
            x = x * kept.float()[:, :, None]

            # Rescale by probability of being kept over n_tests
            proba_kept = torch.zeros(B, C, device=x.device)
            n_tests = 100

            for _ in range(n_tests):
                center_to_ban = torch.rand(2, device=x.device)
                kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
                proba_kept += kept.float() / n_tests

            # Rescale by inverse probability of being kept
            x = x / (1e-8 + proba_kept[:, :, None])
        return x


class ConvSequence(nn.Module):

    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: tp.Optional[int] = None,
        stride: int = 2,
        dropout: float = 0.0,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0,
        glu: int = 0,
        activation: tp.Any = None,
    ) -> None:
        """
        Convolutional sequence with optional skip connections and GLU activations.

        Arguments:
            channels -- List of channel dims for each layer.

        Keyword Arguments:
            kernel -- Convolutional kernel size (default: {4})
            dilation_growth -- Growth factor for dilation (default: {1})
            dilation_period -- Period for resetting dilation (default: {None})
            stride -- Convolutional stride (default: {2})
            dropout -- Dropout rate (default: {0.0})
            decode -- If True, uses ConvTranspose1d (default: {False})
            batch_norm -- If True, uses batch normalization (default: {False})
            dropout_input -- Dropout rate for input (default: {0})
            glu -- If > 0, uses GLU activation every `glu` layers (default: {0})
            activation -- Activation function (default: {None})
        """

        super().__init__()

        dilation = 1
        channels = tuple(channels)
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()

        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d

        # Build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Add input dropout if specified for first layer
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # Add convolutional layer with optional dilation
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1

            pad = kernel // 2 * dilation

            layers.append(
                Conv(
                    chin,
                    chout,
                    kernel,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=1,
                )
            )

            dilation *= dilation_growth

            # Batch norm, activation, and dropout
            if not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())

                if dropout:
                    layers.append(nn.Dropout(dropout))

            self.sequence.append(nn.Sequential(*layers))

            # Add GLU layer if specified
            if glu and (k + 1) % glu == 0:
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, chout * 2, 1 + 2, padding=1),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):

            old_x = x
            x = module(x)

            # Residual
            if x.shape == old_x.shape:
                x = x + old_x

            glu = self.glus[module_idx]

            if glu is not None:
                x = glu(x)

        return x
