# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import torch
from torch import nn
import mne
from studies.study import Recording


class PositionGetter:
    # definitely out of range regardless of scaling type
    INVALID = -2.0

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
                        for i in range(len(recording.info["chs"]))
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
    """
    Spatial dropout by random center and radius (self.dropout) in normalized [0, 1]
    coordinates, where all within the radius in 2 or 3D space is zeroed out.
    """

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
        positions = self.position_getter.get_positions(x, recording)  # [B, C, dim]
        valid_mask = ~self.position_getter.is_invalid(positions)  # [B, C]

        x = x * valid_mask.unsqueeze(-1)  # [B, C, 1]

        if self.training:

            # Spatial dropout by random center and radius
            center_to_ban = torch.rand(self.position_getter.dim, device=x.device)
            dist = (positions - center_to_ban).norm(dim=-1)

            # Keep channels outside the dropout radius or invalid
            kept = (dist > self.dropout) | ~valid_mask  # [B, C]

            # Zero out channels for dropout
            x = x * kept.unsqueeze(-1).float()

            # Rescale by probability of being kept over n_tests
            n_tests = 100
            prob_kept = torch.zeros_like(kept, dtype=torch.float)

            for _ in range(n_tests):
                center_to_ban = torch.rand(self.position_getter.dim, device=x.device)
                dist = (positions - center_to_ban).norm(dim=-1)
                tmp_kept = (dist > self.dropout) | ~valid_mask
                prob_kept += tmp_kept.float() / n_tests

            # Rescale by inverse probability of being kept
            x = x / (prob_kept.unsqueeze(-1) + 1e-8)

        return x


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
            torch.empty((len(conditions), in_channels, out_channels)),
        )
        nn.init.kaiming_uniform_(self.weights, a=0)

    @property
    def trained_indices_list(self):
        return list(self.trained_indices)

    def forward(
        self,
        x: torch.Tensor,
        condition: tp.Union[str, torch.LongTensor],
    ) -> torch.Tensor:
        """Applies a conditional linear transformation to the input tensor.

        Arguments:
            x -- input tensor with shape [B, C, T]
            condition -- string specifying condition or selected indices of shape [B]
                This is different to channel merger as merging occurs before concatenation.

        Returns:
            torch.Tensor -- output tensor with shape [B, C, T]
        """
        S, C, D = self.weights.shape
        B, C, T = x.shape

        # If a single condition string, fallback to old logic:
        if isinstance(condition, str):
            if condition not in self.conditions:
                index = self.conditions["unknown"]
            else:
                index = self.conditions[condition]
                if self.training:
                    self.trained_indices.add(index)
            w = self.weights[index]  # [C, D]
            # [B, C, T] x [C, D] => [B, D, T]
            return torch.einsum("bct,cd->bdt", x, w)

        # Otherwise, we assume condition is a LongTensor of shape [B]
        assert (
            condition.dim() == 1 and condition.shape[0] == B
        ), f"condition indices must be [B], got {condition.shape}"

        # Add to trained indices
        if self.training:
            unique_ids = condition.unique()
            for idx in unique_ids.tolist():
                self.trained_indices.add(int(idx))

        W = self.weights[condition]  # [B, C, D]
        return torch.einsum("bct,bcd->bdt", x, W)  # [B, D, T]

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
        stride: int = 1,
        dropout: float = 0.0,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0,
        glu: int = 0,
        activation: tp.Any = None,
        half: bool = False,
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
            half -- If True, uses stride 2 for third to last layer (default: {False})
                This downsamples the input by 2x.
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

            conv_layer = Conv(
                chin,
                chout,
                kernel * 2 - 1 if (half and (k <= len(channels) - 4)) else kernel,
                stride * 2 if (half and (k == len(channels) - 4)) else stride,
                pad,
                dilation=dilation,
                groups=1,
            )

            nn.init.kaiming_uniform_(conv_layer.weight, a=0)
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)

            layers.append(conv_layer)

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
                glu_layer = nn.Sequential(
                    nn.Conv1d(
                        in_channels=chout,
                        out_channels=chout * 2,
                        kernel_size=1 + 2,
                        padding=1,
                    ),
                    nn.GLU(dim=1),
                )
                nn.init.kaiming_uniform_(glu_layer[0].weight, a=0)
                self.glus.append(glu_layer)

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
