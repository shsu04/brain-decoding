from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np
import itertools
import functools
from torch import Tensor
from scipy.special import sph_harm


class TemplateEmbedding(nn.Module):
    """
    Learns optimal common spatial representations across different input layout
    to produce high dim embeddings.
    """

    def __init__(
        self, dimension: int = 2048, n_template_points: int = 64, rbf_sigma: float = 0.1
    ):
        super().__init__()
        # Leaned template positions
        self.template_positions = nn.Parameter(torch.randn(n_template_points, 2))
        # Projects RBF weights to high-dimensional space
        self.embedding = nn.Linear(n_template_points, dimension)
        self.rbf_sigma = rbf_sigma

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: Channel positions [B, C, dims]
        Returns:
            embeddings: Position embeddings [B, C, D]
        """
        # Compute distances to template positions
        dists = torch.cdist(positions, self.template_positions[None])

        # RBF interpolation weights
        weights = torch.exp(-(dists**2) / (2 * self.rbf_sigma**2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Project to embedding space
        return self.embedding(weights)


class SphericalEmbedding(nn.Module):
    """
    Projects positions onto a unit sphere and computes spherical harmonic
    coefficients. Learns optimal weighting of harmonic components for
    embedding positionsonto a high-dimensional space.
    """

    def __init__(self, dimension: int = 2048, max_degree: int = 8):
        """
        Args:
            dimension (int): Output dimension of the embedding.
            max_degree (int): Maximum degree of the spherical harmonics.
        """
        super().__init__()
        self.max_degree = max_degree
        self.dimension = dimension

        # Number of spherical harmonic components
        n_harmonics = (max_degree + 1) ** 2

        # Learnable weights for each harmonic component
        self.harmonics_weights = nn.Parameter(torch.randn(n_harmonics, dimension))

    def _spherical_harmonic(self, l: int, m: int, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Computes spherical harmonic Y_l^m(theta, phi) for given angles.

        Args:
            l (int): Degree of the spherical harmonic.
            m (int): Order of the spherical harmonic.
            theta (Tensor): Polar angle (in radians) [batch, channels].
            phi (Tensor): Azimuthal angle (in radians) [batch, channels].

        Returns:
            Tensor: Real part of the spherical harmonic Y_l^m [batch, channels].
        """
        # Compute spherical harmonics using scipy's sph_harm
        Y_lm = np.vectorize(lambda t, p: sph_harm(m, l, p, t), otypes=[np.complex128])
        theta_np, phi_np = theta.cpu().numpy(), phi.cpu().numpy()
        Y = torch.from_numpy(Y_lm(theta_np, phi_np).real).to(theta.device)
        return Y

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: Positions in 3D space [batch, channels, 3].

        Returns:
            embeddings: Spherical harmonic embeddings [batch, channels, dimension].
        """
        # Normalize positions to project onto the unit sphere
        r = torch.norm(positions, dim=-1, keepdim=True)
        positions = positions / (r + 1e-8)

        # Convert to spherical coordinates
        theta = torch.acos(
            torch.clamp(positions[..., 2], -1.0, 1.0)
        )  # Clamping for numerical stability
        phi = torch.atan2(positions[..., 1], positions[..., 0])

        # Compute spherical harmonics for each (l, m) pair
        harmonics = []
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                Y = self._spherical_harmonic(l, m, theta, phi)
                harmonics.append(Y)

        harmonics = torch.stack(harmonics, dim=-1)  # [batch, channels, n_harmonics]

        # Compute weighted sum of harmonics
        embeddings = torch.matmul(
            harmonics, self.harmonics_weights
        )  # [batch, channels, dimension]

        return embeddings


class AdaptiveGridMerger(nn.Module):
    """
    Learns a common grid representation across different channel layouts using
    soft assignments to grid points.

    Key advantages:
    - Explicit spatial structure preservation
    - Interpretable intermediate representation
    - Handles arbitrary input layouts through interpolation
    - Good for visualization and understanding learned patterns

    The approach:
    1. Defines a regular grid in space
    2. Softly assigns input channels to grid points
    3. Learns optimal mapping from grid to output channels
    """

    def __init__(
        self,
        merger_channels: int,
        grid_size: Tuple[int, ...] = (8, 8),
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.smoothing = smoothing
        # Learnable mapping from grid points to output channels
        self.grid_weights = nn.Parameter(
            torch.randn(merger_channels, np.prod(grid_size))
        )

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            x: Input features [batch, channels, time]
            positions: Channel positions [batch, channels, dims]
        Returns:
            merged: Merged channel outputs [batch, merger_channels, time]
        """
        # Scale positions to grid coordinates
        grid_pos = (positions + 1) * torch.tensor(self.grid_size) / 2

        # Compute interpolation weights for each dimension
        indices = []
        weights = []
        for dim in range(len(self.grid_size)):
            pos = grid_pos[..., dim]
            idx_low = pos.floor().long()
            idx_high = pos.ceil().long()
            w_high = pos - idx_low
            w_low = 1 - w_high

            indices.append((idx_low, idx_high))
            weights.append((w_low, w_high))

        # Distribute channel values to grid points using trilinear interpolation
        grid_values = torch.zeros(
            *x.shape[:-2], np.prod(self.grid_size), device=x.device
        )

        for idx_combo in itertools.product(*[(0, 1)] * len(self.grid_size)):
            idx = [indices[d][i] for d, i in enumerate(idx_combo)]
            w = [weights[d][i] for d, i in enumerate(idx_combo)]
            weight = functools.reduce(lambda a, b: a * b, w)

            # Convert multidimensional index to flat index
            grid_idx = sum(
                idx[d] * np.prod(self.grid_size[d + 1 :])
                for d in range(len(self.grid_size))
            )

            grid_values.scatter_add_(-1, grid_idx, x * weight.unsqueeze(-1))

        # Map grid to output channels
        return torch.matmul(self.grid_weights, grid_values)


class MultiResolutionMerger(nn.Module):
    """
    Merges channels using multiple spatial scales, adapting to both local and
    global patterns in the data.

    Key advantages:
    - Captures patterns at multiple spatial scales
    - Adaptively determines optimal scale per region
    - Robust to layout differences and sensor density variations
    - Good for hierarchical spatial patterns

    The approach:
    1. Computes channel relationships at multiple spatial scales
    2. Learns optimal scale combination for each output channel
    3. Allows different regions to use different dominant scales
    """

    def __init__(
        self, merger_channels: int, n_scales: int = 3, base_sigma: float = 0.1
    ):
        super().__init__()
        # Generate geometric sequence of spatial scales
        self.scales = [(base_sigma * (2**i)) for i in range(n_scales)]
        self.merger_channels = merger_channels

        # Network to learn optimal scale combination
        self.weight_net = nn.Sequential(
            nn.Linear(n_scales, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # Learnable target positions for output channels
        self.target_positions = nn.Parameter(torch.randn(merger_channels, 2))

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            x: Input features [batch, channels, time]
            positions: Channel positions [batch, channels, dims]
        Returns:
            merged: Merged channel outputs [batch, merger_channels, time]
        """
        # Compute distances to target positions
        dists = torch.cdist(positions, self.target_positions[None])

        # Calculate weights at each spatial scale
        scale_weights = []
        for sigma in self.scales:
            weights = torch.exp(-(dists**2) / (2 * sigma**2))
            scale_weights.append(weights)

        # Learn optimal scale combination
        scale_weights = torch.stack(scale_weights, dim=-1)
        weights = self.weight_net(scale_weights).squeeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Apply weights to input features
        return torch.einsum("bct,bcm->bmt", x, weights)


class EquivariantMerger(nn.Module):
    """
    Merges channels using operations that are equivariant to rotations and
    translations of the input layout.

    Key advantages:
    - Preserves geometric relationships under transformations
    - Generalizes well across different recording setups
    - Theoretically well-founded geometric deep learning approach
    - Robust to layout variations and rotations

    The approach:
    1. Uses local coordinate frames to achieve equivariance
    2. Applies series of equivariant transformations
    3. Projects to output channels while maintaining equivariance
    """

    def __init__(self, merger_channels: int, n_layers: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EquivariantLayer(
                    in_dim=2 if i == 0 else hidden_dim,  # or 3 for 3D
                    out_dim=hidden_dim,
                )
                for i in range(n_layers)
            ]
        )
        self.output_heads = nn.Parameter(torch.randn(merger_channels, hidden_dim))

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            x: Input features [batch, channels, time]
            positions: Channel positions [batch, channels, dims]
        Returns:
            merged: Merged channel outputs [batch, merger_channels, time]
        """
        # Apply sequence of equivariant transformations
        features = x
        for layer in self.layers:
            features = layer(features, positions)

        # Project to output channels while maintaining equivariance
        return torch.matmul(self.output_heads, features.transpose(-1, -2))


class EquivariantLayer(nn.Module):
    """Helper class for equivariant operations"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Applies equivariant transformation to input features
        """
        # Build local coordinate frames
        dists = torch.cdist(positions, positions)
        weights = torch.exp(-(dists**2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Transform features in local coordinates
        return self.linear(torch.matmul(weights, x))
