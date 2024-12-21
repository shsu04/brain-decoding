from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from scipy.special import sph_harm
import math

from .common import PositionGetter
from studies import Recording


class ChannelMerger(nn.Module):
    def __init__(
        self,
        merger_channels: int,
        embedding_type="fourier",
        embedding_dim: int = 256,
        dropout: float = 0.2,
        conditions: dict[str, int] = None,
        layout_dim: int = 2,
        layout_proj: bool = False,
        layout_scaling: str = "midpoint",
    ):
        super().__init__()

        self.merger_channels = merger_channels
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.position_getter = PositionGetter(
            dim=layout_dim, proj=layout_proj, scaling=layout_scaling
        )

        assert embedding_dim % 4 == 0

        # EMBEDDING
        self.embedding_type = embedding_type
        self.embedding_type = embedding_type
        if embedding_type == "fourier":
            assert layout_dim == 2 and layout_scaling == "minmax"
            self.embedding = FourierEmbedding(dimension=embedding_dim)
        elif embedding_type == "spherical":
            assert layout_dim == 3 and layout_scaling == "midpoint"
            self.embedding = SphericalEmbedding(dimension=embedding_dim)
        elif embedding_type == "linear":
            self.embedding = nn.Sequential(
                nn.Linear(layout_dim, embedding_dim), nn.Tanh()
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # Learnable heads for each condition
        self.conditions = conditions
        if self.conditions:
            assert "unknown" in conditions, "Conditions must include an 'unknown' key"
            self.trained_indices = set()
            self.heads = nn.Parameter(
                torch.empty(
                    (
                        len(self.conditions),
                        merger_channels,
                        embedding_dim,
                    )
                )
            )
        else:
            self.heads = nn.Parameter(torch.empty((merger_channels, embedding_dim)))
        nn.init.kaiming_uniform_(self.heads, a=0)

    @property
    def trained_indices_list(self):
        return list(self.trained_indices)

    def get_heads(self, B: int, condition: str = None, device: str = "cpu"):

        if not self.conditions:
            # [ch_out, dim] -> [B, ch_out, dim]
            heads = self.heads[None].expand(B, -1, -1)

        # If conditional, choose heads based on condition
        else:
            _, chout, pos_dim = self.heads.shape

            # Take mean of trained indices
            if condition == "mean" and len(self.trained_indices) > 0:
                # [B, ch_out, emb_dim]
                heads = (
                    torch.stack([self.heads[self.trained_indices_list]])
                    .mean(dim=0)
                    .expand(B, -1, -1)
                    .to(device)
                )
            else:
                # Expand head to shape B
                if condition not in self.conditions.keys():
                    index = self.conditions["unknown"]
                else:
                    index = self.conditions[condition]
                    if self.training:
                        self.trained_indices.add(index)

                # [1, ch_out, emb_dim] -> [B, ch_out, emb_dim]
                conditions = torch.full((B,), index, dtype=torch.long).to(device)
                heads = self.heads.gather(
                    0, conditions.view(-1, 1, 1).expand(-1, chout, pos_dim)
                )
        return heads

    def forward(self, x: torch.Tensor, recording: Recording, condition: str = None):
        """
        Arguments:
            x -- input tensor with shape [B, C, T]
            recording -- recording object with channel layout
            condition -- condition to select heads from. Can also be "mean"

        Returns:
            torch.Tensor -- output tensor with shape [B, merger_channels, T]
        """
        B, C, T = x.shape

        positions = self.position_getter.get_positions(x, recording)  # [B, C, 2]

        # Mask invalid channels, [B, C]
        score_offset = torch.zeros(B, C, device=x.device)
        score_offset[self.position_getter.is_invalid(positions)] = float("-inf")

        # Spatial embedding, [B, C, dim] -> [B, C, emb_dim]
        embedding = self.embedding(positions)

        # Dropout around random center's radius
        if self.training and self.dropout:

            center_to_ban = torch.rand(self.position_getter.dim, device=x.device)
            radius_to_ban = self.dropout

            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float("-inf")  # [B, C]

        heads = self.get_heads(B=B, condition=condition, device=x.device)

        # How well pos emb aligns with learnable heads
        scores = torch.einsum("bcd,bod->boc", embedding, heads)  # [B, C, ch_out]
        scores += score_offset[:, None]  # mask

        # Create each output channel as a weighted sum of input channels
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", x, weights)  # [B, ch_out, T]

        return out


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
