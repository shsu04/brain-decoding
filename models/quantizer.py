import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Quantizer(nn.Module):
    def __init__(self, dim: int, num_codebooks: int, codebook_size: int):
        """
        Args:
            dim: Dimension of input vectors and codebook entries
            num_codebooks: Number of separate codebooks
            codebook_size: Size of each codebook
        """
        super().__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input tensor of shape [B, dim, T]
        Returns:
            quantized: Quantized tensor of shape [B, dim, T]
            metrics: Dictionary containing relevant metrics
        """
        raise NotImplementedError


class VQQuantizer(Quantizer):
    def __init__(
        self, dim: int, num_codebooks: int, codebook_size: int, commitment: float = 0.25
    ):
        super().__init__(dim, num_codebooks, codebook_size)
        self.commitment = commitment

        # Codebook entry has dimension dim/num_codebooks
        assert dim % num_codebooks == 0, "dim must be divisible by num_codebooks"
        self.chunk_size = dim // num_codebooks

        # [n_codebooks, codebook_size, chunk_size]
        self.codebooks = nn.Parameter(
            torch.empty((num_codebooks, codebook_size, self.chunk_size)),
        )
        nn.init.kaiming_uniform_(self.codebooks, a=0)

        print(
            f"VQQuantizer initialized with {num_codebooks} codebooks of size {codebook_size}"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Vector quantization with straight-through gradients and commitment loss

        ||x - c||^2 = ||x||^2 + ||c||^2 - 2x^T c

        Args:
            x: Input tensor [B, dim, T]

        Returns:
            quantized: Quantized tensor [B, dim, T]
            metrics: Contains 'commit_loss' and 'perplexity' per codebook
        """

        # [B, D, T] -> [B, n_cbooks, chunk_size, T], since D = n_cbooks * chunk_size
        B, D, T = x.shape
        assert (
            D == self.dim
        ), f"Input dimension {D} does not match quantizer dimension {self.dim}"

        x_chunks = x.view(B, self.num_codebooks, self.chunk_size, T)

        # ||x||^2
        x_norm = x_chunks.pow(2).unsqueeze(2)  # [B, n_cbooks, 1, chunk_size, T]
        x_norm = x_norm.sum(dim=3)  # [B, n_cbooks, 1, T]

        # ||c||^2 term
        c_norm = self.codebooks.pow(2).sum(2)  # [n_cbooks, cbook_size]
        c_norm = c_norm.unsqueeze(0).unsqueeze(-1)  # [1, n_cbooks, cbook_size, 1]
        c_norm = c_norm.expand(B, -1, -1, T)  # [B, n_cbooks, cbook_size, T]

        # -2x^T c term
        cross_term = -2 * torch.einsum(
            "bkdt,kcd->bkct", x_chunks, self.codebooks
        )  # [B, n_cbooks, cbook_size, T]

        # Closest codebook entries
        distances = x_norm + c_norm + cross_term  # [B, n_cbooks, codebook_size, T]
        indices = distances.argmin(dim=2)  # [B, n_cbooks, T]

        # [B, n_cbooks, chunk_size, T]
        quantized = self.codebooks[
            torch.arange(self.num_codebooks).to(x.device).view(1, -1, 1),  # Codebook
            indices.to(x.device),  # Codebook entry
        ]

        # Metrics
        avg_probs = torch.zeros(self.num_codebooks, self.codebook_size, device=x.device)

        for k in range(self.num_codebooks):
            # occurrences of each index, [codebook_size]
            avg_probs[k] = torch.bincount(
                indices[:, k].flatten(),  # [B, T]
                minlength=self.codebook_size,
            ).float() / (
                B * T
            )  # Normalize

        perplexity = torch.exp(
            torch.nansum(-avg_probs * torch.log(avg_probs + 1e-8), dim=1)
        )  # [n_codebooks]

        codebook_loss = F.mse_loss(x_chunks.detach(), quantized)
        encoder_loss = F.mse_loss(x_chunks, quantized.detach())
        commit_loss = codebook_loss + self.commitment * encoder_loss

        # Forward uses quantized, backward uses gradient = 1
        quantized = x_chunks + (quantized - x_chunks).detach()

        return (
            quantized.reshape(B, -1, T),  # [B, dim, T]
            {
                "commit_loss": commit_loss,
                "perplexity": perplexity,
            },
        )


class GumbelQuantizer(Quantizer):
    def __init__(
        self,
        dim: int,
        num_codebooks: int,
        codebook_size: int,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
        temp_decay: float = 0.999,
    ):
        super().__init__(dim, num_codebooks, codebook_size)

        # Temperature annealing for stability
        self.register_buffer("temp", torch.tensor(temp_init))
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        assert dim % num_codebooks == 0, "dim must be divisible by num_codebooks"
        self.chunk_size = dim // num_codebooks

        # Projection to logits
        self.projections = nn.ModuleList(
            [nn.Conv1d(self.chunk_size, codebook_size, 1) for _ in range(num_codebooks)]
        )
        nn.init.kaiming_uniform_(self.projections, a=0)

        # [n_codebooks, codebook_size, chunk_size]
        self.codebooks = nn.Parameter(
            torch.empty((num_codebooks, codebook_size, self.chunk_size))
        )
        nn.init.kaiming_uniform_(self.codebooks, a=0)

        print(
            f"GumbelQuantizer initialized with {num_codebooks} codebooks of size {codebook_size}"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Gumbel-softmax quantization with learned codebooks

        Args:
            x: Input tensor [B, dim, T]

        Returns:
            quantized: Quantized tensor [B, dim, T]
            metrics: Contains 'temp' and 'perplexity' per codebook
        """
        B, D, T = x.shape
        assert (
            D == self.dim
        ), f"Input dimension {D} does not match quantizer dimension {self.dim}"

        # [B, D, T] -> [B, n_cbooks, chunk_size, T]
        x_chunks = x.view(B, self.num_codebooks, self.chunk_size, T)

        quantized_chunks, perplexity = [], []

        for k in range(self.num_codebooks):

            logits = self.projections[k](x_chunks[:, k])  # [B, codebook_size, T]

            # Gumbel softmax sampling
            soft_onehot = F.gumbel_softmax(
                logits.transpose(1, 2),  # [B, T, codebook_size]
                tau=self.temp,
                hard=True,
            ).transpose(
                1, 2
            )  # [B, codebook_size, T]

            # Perplexity
            avg_probs = soft_onehot.mean(dim=(0, 2))  # [codebook_size]
            perplexity.append(
                torch.exp(torch.nansum(-avg_probs * torch.log(avg_probs + 1e-8), dim=0))
            )  # [codebook_size]

            # [B, chunk_size, T]
            quantized = torch.einsum("bct,kc->bkt", soft_onehot, self.codebooks[k])
            quantized_chunks.append(quantized)

        # Update temp
        if self.training:
            self.temp.data.mul_(self.temp_decay).clamp_(min=self.temp_min)

        # [B,dim / cbook_size, T] -> [B, dim, T]
        quantized = torch.cat(quantized_chunks, dim=1)

        return quantized, {"temp": self.temp, "perplexity": torch.stack(perplexity)}
