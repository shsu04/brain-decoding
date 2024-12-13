# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp
import torch
from torch import nn
import mne
from studies.study import Recording


class PositionGetter:
    INVALID = -0.1

    def __init__(
        self, dim: int = None, proj: bool = False, scaling: str = "midpoint"
    ) -> None:
        self.dim = dim
        self.proj = proj
        self.scaling = scaling
        self._cache: tp.Dict[int, torch.Tensor] = {}

        assert dim in [2, 3, None], f"Layout can only be 2D or 3D. Invalid dim {dim}."
        assert scaling in [
            "midpoint",
            "minmax",
            "standard",
            "maxabs",
        ], f"Invalid scaling type {scaling}"
        assert not (proj and dim == 3), "Cannot project to 3D"

    def load_sensor_layout(self, recording: Recording) -> torch.Tensor:
        """
        Returns the scaled sensor locations of the neural recording.
        Channels already valid from since picked in load raw in Recording

        If proj, 3D layout is projected to 2D space.

        Arguments:
            scaling -- type of scaling to apply, can be:
                "midpoint" - scale to [-1, 1] around geometric midpoint
                "minmax" - scale to [0, 1] based on min and max
                "standard" - scale to mean 0 and std 1
                "maxabs" - scale to [-1, 1] based on max absolute value

        Returns:
            if dim == 2:
                scaled positions of the sensors in 2D space. Dim = [C, 2], (x, y)
            if dim == 3:
                scaled positions of the sensors in 3D space. Dim = [C, 3], (x, y, z)
        """
        if recording.cache_path in self._cache:
            return self._cache[recording.cache_path]

        # Get layout
        try:
            assert (
                recording.info is not None
            ), f"Recording info is not available. {recording.cache_path}"

            # Find layout projects 3D to 2D
            if self.proj:
                layout = mne.find_layout(
                    recording.info,
                )
                layout = torch.tensor(
                    layout.pos[: len(recording.channel_names), :2], dtype=torch.float32
                )  # [C, 2]
            # Get x, y, z coordinates without projection
            else:
                layout = torch.tensor(
                    [
                        recording.info["chs"][i]["loc"][: self.dim]
                        for i in range(len(self.info["chs"]))
                        if recording.info["ch_names"][i] in recording.channel_names
                    ],
                    dtype=torch.float32,
                )  # [C, dim]
        except Exception as e:
            raise ValueError(f"Error loading layout for {recording.cache_path}. {e}")

        # Scaling each dim independently
        if self.scaling == "midpoint":
            midpoints = layout.mean(dim=0)
            max_deviation = (layout - midpoints).abs().max(dim=0).values
            layout = (layout - midpoints) / max_deviation

        elif self.scaling == "minmax":
            mins = layout.min(dim=0).values
            maxs = layout.max(dim=0).values
            layout = (layout - mins) / (maxs - mins)

        elif self.scaling == "standard":
            means = layout.mean(dim=0)
            stds = layout.std(dim=0)
            layout = (layout - means) / stds

        elif self.scaling == "maxabs":
            max_abs = layout.abs().max(dim=0).values
            layout = layout / max_abs

        self._cache[recording.cache_path] = layout

        return layout

    def get_positions(self, x: torch.Tensor, recording: Recording) -> torch.Tensor:
        """Returns scaled positions of channels in x.

        Arguments:
            x -- recordings with shape [B, C, T]

        Returns:
            torch.Tensor -- positions of channels in x [B, C, dim]
        """
        B, C, T = x.shape
        positions = torch.full((B, C, self.dim), self.INVALID, device=x.device)
        layout = self.load_sensor_layout(recording)

        for idx in range(len(x)):
            positions[idx, : len(layout)] = layout.to(x.device)

        return positions

    def is_invalid(self, positions: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask of invalid channels.

        Arguments:
            positions -- 2D positions of channels in x [B, C, dim]

        Returns:
            torch.Tensor -- boolean mask of invalid channels [B, C]
        """

        return (positions == self.INVALID).all(dim=-1)


class ChannelDropout(nn.Module):
    """Spatial dropout by random center and radius in normalized [0, 1] coordinates."""

    def __init__(
        self,
        dropout: float = 0.1,
        layout_dim: int = 2,
        layout_proj: bool = False,
        layout_scaling: str = "midpoint",
    ):
        super().__init__()
        self.dropout = dropout
        self.position_getter = PositionGetter(
            dim=layout_dim, proj=layout_proj, scaling=layout_scaling
        )

    def forward(self, x: torch.Tensor, recording: Recording) -> torch.Tensor:
        """
        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
        """

        if not self.dropout:
            return x

        B, C, T = x.shape

        # Mask out invalid channels
        positions = self.position_getter.get_positions(x, recording)
        valid = (~self.position_getter.is_invalid(positions)).float()

        x = x * valid[:, :, None]

        if self.training:

            # Spatial dropout by random center and radius
            center_to_ban = torch.rand(self.position_getter.dim, device=x.device)
            kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
            x = x * kept.float()[:, :, None]

            # Rescale by probability of being kept over n_tests
            proba_kept = torch.zeros(B, C, device=x.device)
            n_tests = 100

            for _ in range(n_tests):
                center_to_ban = torch.rand(self.position_getter.dim, device=x.device)
                kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
                proba_kept += kept.float() / n_tests

            # Rescale by inverse probability of being kept
            x = x / (1e-8 + proba_kept[:, :, None])
        return x


class FourierEmbedding(nn.Module):
    """
    Fourier positional embedding. Maps each channel to a high dimensional space
    through a learnt channel-specific function over sensor locations.
    Unlike trad. embedding this is not using exponential periods for cos and sin,
    but typical `2 pi k` which can represent any function over [0, 1]. As this
    function would be necessarily periodic, we take a bit of margin and do over
    [-0.2, 1.2].
    """

    def __init__(self, dimension: int = 2048, margin: float = 0.2):
        super().__init__()

        # Grid size. e.g. dim=2048, n_freqs=32
        n_freqs = (dimension // 2) ** 0.5
        assert int(n_freqs**2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
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


class ChannelMerger(nn.Module):
    """
    Merges channels by learning a weighted sum of input channels, where the
    weights are determined by the alignment of the channel's positial embeddings
    with learnable heads.
    """

    def __init__(
        self,
        merger_channels: int,
        embedding_dim: int = 2048,
        dropout: float = 0,
        conditions: dict[str, int] = None,
        layout_dim: int = 2,
        layout_proj: bool = False,
        layout_scaling: str = "midpoint",
    ):
        super().__init__()
        assert embedding_dim % 4 == 0

        self.position_getter = PositionGetter(
            dim=layout_dim, proj=layout_proj, scaling=layout_scaling
        )

        # Learnable heads for each condition
        self.conditions = conditions
        if self.conditions:
            assert "unknown" in conditions, "Conditions must include an 'unknown' key"
            self.trained_indices = set()
            self.heads = nn.Parameter(
                torch.randn(
                    len(self.conditions),
                    merger_channels,
                    embedding_dim,
                    requires_grad=True,
                )
            )
        else:
            self.heads = nn.Parameter(
                torch.randn(merger_channels, embedding_dim, requires_grad=True)
            )

        self.heads.data /= embedding_dim**0.5
        self.dropout = dropout
        self.embedding = FourierEmbedding(dimension=embedding_dim)

    @property
    def trained_indices_list(self):
        return list(self.trained_indices)

    def forward(
        self, x: torch.Tensor, recording: Recording, condition: str = None
    ) -> torch.Tensor:
        """
        Arguments:
            x -- input tensor with shape [B, C, T]
            recording -- recording object with channel layout
            condition -- condition to select heads from. Can also be "mean"

        Returns:
            torch.Tensor -- output tensor with shape [B, merger_channels, T]
        """

        B, C, T = x.shape

        # Spatial embedding, [B, C, dim] -> [B, C, emb_dim]
        positions = self.position_getter.get_positions(x, recording)
        embedding = self.embedding(positions)

        # Mask out invalid channels, [B, C]
        score_offset = torch.zeros(B, C, device=x.device)
        score_offset[self.position_getter.is_invalid(positions)] = float("-inf")

        # Dropout by random center and radius
        if self.training and self.dropout:

            center_to_ban = torch.rand(self.position_getter.dim, device=x.device)
            radius_to_ban = self.dropout

            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float("-inf")

        # If conditional, choose heads based on condition
        if self.conditions:

            _, chout, pos_dim = self.heads.shape

            # Take mean of trained indices
            if condition == "mean":
                # [B, ch_out, emb_dim]
                heads = (
                    torch.stack([self.heads[self.trained_indices_list]])
                    .mean(dim=0)
                    .expand(B, -1, -1)
                )
            else:
                # Expand unknown head to shape B
                if condition not in self.conditions:
                    index = self.conditions["unknown"]
                # Expand known head to shape B
                else:
                    index = self.conditions[condition]
                    if self.training:
                        self.trained_indices.add(index)

                # [1, ch_out, emb_dim] -> [B, ch_out, emb_dim]
                conditions = torch.full((B,), index, dtype=torch.long)
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
        conditions: dict[str, int],
    ):
        super().__init__()
        assert "unknown" in conditions, "Conditions must include an 'unknown' key"

        self.conditions = conditions
        self.trained_indices = set()
        self.weights = nn.Parameter(
            torch.randn(len(conditions), in_channels, out_channels)
        )
        self.weights.data *= 1 / in_channels**0.5

    @property
    def trained_indices_list(self):
        return list(self.trained_indices)

    def forward(self, x: torch.Tensor, condition: str) -> torch.Tensor:
        """Applies a conditional linear transformation to the input tensor.

        Arguments:
            x -- input tensor with shape [B, C, T]
            condition -- string specifying condition, can be "mean"

        Returns:
            torch.Tensor -- output tensor with shape [B, C, T]
        """
        S, C, D = self.weights.shape
        B, C, T = x.shape

        if condition == "mean":
            weights = (
                torch.stack([self.weights[self.trained_indices_list]])
                .mean(dim=0)
                .expand(B, -1, -1)
            )
        else:
            # Expand unknown cond to shape B
            if condition not in self.conditions:
                index = self.conditions["unknown"]
            # Expand known head to shape B
            else:
                index = self.conditions[condition]
                if self.training:
                    self.trained_indices.add(index)

            # Gather weight matrices for each condition
            conditions = torch.full((B,), index, dtype=torch.long)
            weights = self.weights.gather(0, conditions.view(-1, 1, 1).expand(-1, C, D))

        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"Condition layers({C}, {D}, {S})"


class ConvSequence(nn.Module):
    """Convolutional sequence with optional skip connections and GLU activations."""

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
