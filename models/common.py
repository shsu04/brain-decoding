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


class ScaledEmbedding(nn.Module):
    """Scale up learning rate for the embedding, otherwise, it can move too slowly."""

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


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


class DualPathRNN(nn.Module):
    def __init__(self, channels: int, depth: int, inner_length: int = 10):
        super().__init__()
        self.lstms = nn.ModuleList(
            [nn.LSTM(channels, channels, 1) for _ in range(depth * 4)]
        )
        self.inner_length = inner_length

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        IL = self.inner_length

        x = pad_multiple(x, self.inner_length)
        x = x.permute(2, 0, 1).contiguous()  # [L, B, C]

        for idx, lstm in enumerate(self.lstms):

            # Reshape for local/global processing
            y = x.reshape(-1, IL, B, C)

            if idx % 2 == 0:
                # Local processing on chunks
                y = y.transpose(0, 1).reshape(IL, -1, C)
            else:
                # Global processing on across chunks
                y = y.reshape(-1, IL * B, C)

            y, _ = lstm(x)
            if idx % 2 == 0:
                y = y.reshape(IL, -1, B, C).transpose(0, 1).reshape(-1, B, C)
            else:
                y = y.reshape(-1, B, C)
            x = x + y

            if idx % 2 == 1:
                x = x.flip(dims=(0,))
        return x[:L].permute(1, 2, 0).contiguous()


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
        usage_penalty: float = 0.0,
        n_subjects: int = 200,
        per_subject: bool = False,
    ):
        super().__init__()
        assert embedding_dim % 4 == 0

        self.position_getter = PositionGetter()

        # Learnable heads per subject or shared
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(
                torch.randn(
                    n_subjects, merger_channels, embedding_dim, requires_grad=True
                )
            )
        else:
            self.heads = nn.Parameter(
                torch.randn(merger_channels, embedding_dim, requires_grad=True)
            )

        self.heads.data /= embedding_dim**0.5
        self.dropout = dropout
        self.embedding = FourierEmbedding(dimension=embedding_dim)

        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.0)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        """
        Merges channels by learning a weighted sum of input channels, where the
        weights are determined by the alignment of the channel's positial embeddings
        with learnable heads.

        Arguments:
            x -- input tensor with shape [B, C, T]
            layout -- layout of the channels with shape [C, 2]

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

        # [ch_out, dim] -> [B, ch_out, dim]
        heads = self.heads[None].expand(B, -1, -1)

        # How well pos emb aligns with learnable heads
        scores = torch.einsum("bcd,bod->boc", embedding, heads)  # [B, C, ch_out]
        scores += score_offset[:, None]  # mask

        # Create each output channel as a weighted sum of input channels
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", x, weights)  # [B, ch_out, T]

        # Usage penalty to encourage equal usage of heads
        if self.training and self.usage_penalty > 0.0:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage

        return out


class SubjectLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))

        # Initialize as identity scaled
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]

        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """Applies a subject-specific linear transformation to the input tensor.

        Arguments:
            x -- input tensor with shape [B, C, T]
            subjects -- subject indices with shape [B]

        Returns:
            torch.Tensor -- output tensor with shape [B, C, T]
        """

        # Subjects is a 1 dimensional tensor of subject indices
        _, C, D = self.weights.shape

        # Gather weight matrices for each subject
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class ChannelDropout(nn.Module):
    def __init__(self, dropout: float = 0.1, rescale: bool = True):
        super().__init__()
        self.dropout = dropout
        self.rescale = rescale
        self.position_getter = PositionGetter()

    def forward(self, x: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        """Spatial dropout by random center and radius in normalized [0, 1] coordinates.

        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
            rescale: at valid, rescale all channels.
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
            if self.rescale:
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
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0,
        skip: bool = False,
        scale: tp.Optional[float] = None,
        rewrite: bool = False,
        activation_on_last: bool = True,
        post_skip: bool = False,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        activation: tp.Any = None,
    ) -> None:
        """
        Convolutional sequence with optional skip connections and GLU activations.

        Arguments:
            channels -- List of channel dims for each layer.

        Keyword Arguments:
            kernel -- Kernel size for convolutions
            dilation_growth -- Factor by which dilation increases between layers
            dilation_period -- Reset dilation after this many layers
            stride -- Stride for convolutions
            dropout -- Dropout probability after each layer
            leakiness -- Negative slope for LeakyReLU
            groups -- Number of groups for grouped convolutions
            decode -- If True, use transposed convolutions
            batch_norm -- Whether to use batch normalization
            dropout_input -- Dropout probability for input
            skip -- Enable residual connections
            scale -- Scale factor for LayerScale
            rewrite -- Add additional 1x1 conv layer after each conv
            activation_on_last -- Whether to apply activation on final layer
            post_skip -- Add depthwise conv after skip connections
            glu_context -- _description_ (default: {0})
            glu_glu -- Use GLU vs regular activation for GLU blocks
            activation -- Custom activation function
        """

        super().__init__()

        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()

        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)

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
                    groups=groups if k > 0 else 1,
                )
            )

            dilation *= dilation_growth

            # Batch norm, activation, and dropout
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())

                if dropout:
                    layers.append(nn.Dropout(dropout))
                # Optional 1x1 convolution rewrite layer
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]

            # Skip connection
            if chin == chout and skip:

                if scale is not None:
                    layers.append(LayerScale(chout, scale))

                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))

            # Add GLU layer if specified
            if glu and (k + 1) % glu == 0:

                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context),
                        act,
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):

            old_x = x
            x = module(x)

            if self.skip and x.shape == old_x.shape:
                x = x + old_x

            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x
