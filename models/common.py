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
        num_conditions, in_dim, out_dim = self.weights.shape
        B, in_dim_x, T = x.shape

        assert (
            in_dim == in_dim_x
        ), f"Input channels={in_dim_x} must match in_dim={in_dim} in self.weights."

        # If a single condition string
        if isinstance(condition, str):
            if condition not in self.conditions:
                # Int index, or unknown
                index = self.conditions.get(condition, self.conditions["unknown"])

                if index not in self.trained_indices:
                    # if not empty, set the unknown to the mean of trained layers
                    if self.trained_indices:
                        with torch.no_grad():
                            self.weights[index].data = self.weights[
                                self.trained_indices_list
                            ].mean(dim=0)

                if self.training:
                    self.trained_indices.add(index)

                w = self.weights[index]
                return torch.einsum("bct,cd->bdt", x, w)  # [B, D, T]

        # Otherwise, if input is a LongTensor of shape [B]
        assert (
            condition.dim() == 1 and condition.shape[0] == B
        ), f"Condition indices must be [B], got {condition.shape}"

        # mark trained indices
        if self.training:
            unique_ids = condition.unique()
            for idx in unique_ids.tolist():
                # if not empty, set the unknown to the mean of trained layers
                if idx not in self.trained_indices and len(self.trained_indices) > 0:
                    with torch.no_grad():
                        self.weights[idx].data = self.weights[
                            self.trained_indices_list
                        ].mean(dim=0)
                # add to trained indices
                self.trained_indices.add(int(idx))

        W = self.weights[condition]
        return torch.einsum("bct,bcd->bdt", x, W)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"Condition layers({C}, {D}, {S})"
