import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))

    # def forward(
    #     self,
    #     x_1: torch.Tensor,
    #     x_2: torch.Tensor,
    # ) -> dict[str, float]:
    #     """
    #     Computes CLIP loss on two embeddings, x_1 and x_2. Both of shape [B, C, T]

    #     Returns:
    #         clip_loss: torch.Tensor, shape [B]
    #         metrics: dict[str, float]
    #     """
    #     assert x_1.size() == x_2.size()
    #     B, C, T = x_1.size()
    #     # [B, C, T] -> [B * T, C]
    #     x_1, x_2 = (x_1.reshape(B * T, C), x_2.reshape(B * T, C))

    #     x_1_norm = x_1 / (x_1.norm(dim=1, keepdim=True) + 1e-8)
    #     x_2_norm = x_2 / (x_2.norm(dim=1, keepdim=True) + 1e-8)

    #     logits = x_1_norm @ x_2_norm.transpose(0, 1)  # [B*T, B*T]
    #     logits = logits / self.temperature

    #     # Diagonal targets
    #     targets = torch.arange(B * T, device=x_1.device)  # [B]
    #     probs = F.log_softmax(logits, dim=-1)  # [B]
    #     clip_loss = F.cross_entropy(probs, targets, reduction="mean")

    #     return {
    #         "loss": clip_loss,  # Still on device
    #         "metrics": self.eval_metrics(probs, targets),  # on CPU
    #     }

    def forward(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
    ) -> dict[str, float]:
        """
        Computes CLIP loss on two embeddings, x_1 and x_2. Both of shape [B, C, T]
        Returns:
            clip_loss: torch.Tensor, shape [B]
            metrics: dict[str, float]
        """
        assert x_1.size() == x_2.size()

        inv_norms = 1 / (1e-8 + x_1.norm(dim=(1, 2), p=2))  # [B]

        # Compute similarity, [B, C, T] x [B, C, T] -> [B, B]
        logits = torch.einsum("bct,dct,d->bd", x_1, x_2, inv_norms) / self.temperature

        # Diagonal targets
        targets = torch.arange(x_1.size(0), device=x_1.device)
        probs = F.log_softmax(logits, dim=-1)  # [B]
        clip_loss = F.cross_entropy(probs, targets, reduction="mean")

        return {
            "loss": clip_loss,  # Still on device
            "metrics": self.eval_metrics(probs, targets),  # on CPU
        }

    def eval_metrics(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Gives evaluation metrics using CLIP loss logits. Top % correct.
        Accuracy computed by total correct predictions / total predictions
        outside of this function for precision reasons.

        Brain prediction can be hard to classify, need more relaxed metrics
        to track progress during training.

        Args:
            probs: torch.Tensor, shape [B, B]
            targets: torch.Tensor, shape [B]

        Returns:
            metrics: dict[str, float]
        """

        # Metrics
        batch_size = probs.shape[0]

        # Top 10% correct
        k = min(10, batch_size)
        topk_values, topk_indices = torch.topk(probs, k, dim=-1)
        correct_tensor = topk_indices.eq(
            targets.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top10_correct = correct_tensor.cpu().sum().item()

        # Top 5% correct
        k = min(5, batch_size)
        topk_values, topk_indices = torch.topk(probs, k, dim=-1)
        correct_tensor = topk_indices.eq(
            targets.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top5_correct = correct_tensor.cpu().sum().item()

        # correct
        predicted_labels = torch.argmax(probs, dim=-1)
        correct = (predicted_labels == targets).cpu().sum().item()

        metrics = {
            "correct": correct,
            "top_10_correct": top10_correct,
            "top_5_correct": top5_correct,
        }

        return metrics
