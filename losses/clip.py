import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.7))

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        """
        Computes CLIP loss on two mels, x_1 and x_2. Both of shape [B, C, T]

        Returns:
            clip_loss: torch.Tensor, shape [B]
            metrics: dict[str, float]
        """
        assert x_1.size() == x_2.size()

        inv_norms = 1 / (1e-8 + x_1.norm(dim=(1, 2), p=2))

        # Compute similarity, [B, C, T] x [B, C, T] -> [B, B]
        logits = torch.einsum("bct,dct,d->bd", x_1, x_2, inv_norms) * self.temperature

        # Diagonal targets
        targets = torch.arange(x_1.size(0), device=x_1.device)
        probs = F.log_softmax(logits, dim=-1)  # [B]
        clip_loss = F.cross_entropy(probs, targets, reduction='mean') 

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
        top_10 = max(1, batch_size // 10)
        topk_values, topk_indices = torch.topk(probs, top_10, dim=-1)
        correct_tensor = topk_indices.eq(
            targets.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top10_correct = correct_tensor.cpu().sum().item()

        # Top 5% correct
        top_5 = max(1, batch_size // 20)
        topk_values, topk_indices = torch.topk(probs, top_5, dim=-1)
        correct_tensor = topk_indices.eq(
            targets.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top5_correct = correct_tensor.cpu().sum().item()

        # Top 1% correct
        top_1 = max(1, batch_size // 100)
        topk_values, topk_indices = torch.topk(probs, top_1, dim=-1)
        correct_tensor = topk_indices.eq(
            targets.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top1_correct = correct_tensor.cpu().sum().item()

        # correct
        predicted_labels = torch.argmax(probs, dim=-1)
        correct_predictions = (predicted_labels == targets).cpu().sum().item()
        correct = correct_predictions

        metrics =  {
            "correct": correct,
            "top_10_correct": top10_correct,
            "top_5_correct": top5_correct,
            "top_1_correct": top1_correct,
        }
        for k, v in metrics.items():
            metrics[k] = v / batch_size if batch_size > 0 else 0
            
        return metrics
