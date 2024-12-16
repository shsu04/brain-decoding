import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        """
        Computes CLIP loss on two mels, x_1 and x_2. Both of shape [B, C, T]

        Returns:
            clip_loss: torch.Tensor, shape [B]
            metrics: dict[str, float]
        """
        assert x_1.size() == x_2.size()

        # Normalize across time per mel band
        x_1, x_2 = F.normalize(x_1, dim=2), F.normalize(x_2, dim=2)

        # Compute similarity, [B, C, T] x [B, C, T] -> [B, B]
        logits = torch.einsum("bct,dct->bd", x_1, x_2)

        # Diagonal targets
        targets = torch.arange(x_1.size(0), device=x_1.device)

        # Symmetric loss
        x_1_loss = self.cross_entropy(logits, targets, reduction="mean")  # [B]
        x_2_loss = self.cross_entropy(logits.T, targets.T, reduction="mean")  # [B]
        clip_loss = (x_1_loss + x_2_loss) / 2  # [B]

        return {
            "loss": clip_loss.mean(),  # Still on device
            "metrics": self.eval_metrics(logits),  # on CPU
        }

    def cross_entropy(self, preds, targets, reduction="none"):
        """Computes cross entropy loss between preds and targets.

        Arguments:
            preds -- [B, B] tensor of predictions
            targets -- [B] tensor of targets.

        Keyword Arguments:
            reduction -- "none" or "mean" (default: {"none"})

        Returns:
            loss -- [B] tensor of losses
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = -log_softmax(preds)[torch.arange(preds.shape[0]), targets]
        return loss.mean() if reduction == "mean" else loss

    def eval_metrics(
        self,
        logits: torch.Tensor,
    ) -> dict[str, float]:
        """
        Gives evaluation metrics using CLIP loss logits. Top % correct.
        Accuracy computed by total correct predictions / total predictions
        outside of this function for precision reasons.

        Brain prediction can be hard to classify, need more relaxed metrics
        to track progress during training.

        Args:
            logits: torch.Tensor, shape [B, B]

        Returns:
            metrics: dict[str, float]
        """

        # Metrics
        total = logits.shape[0]
        score = F.softmax(logits, dim=-1)
        labels = torch.arange(total).to(logits.device)

        # Top 10% correct
        top_10 = max(1, total // 10)
        topk_values, topk_indices = torch.topk(score, top_10, dim=-1)
        correct_tensor = topk_indices.eq(
            labels.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top10_correct = correct_tensor.cpu().sum().item()

        # Top 5% correct
        top_5 = max(1, total // 20)
        topk_values, topk_indices = torch.topk(score, top_5, dim=-1)
        correct_tensor = topk_indices.eq(
            labels.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top5_correct = correct_tensor.cpu().sum().item()

        # Top 1% correct
        top_1 = max(1, total // 100)
        topk_values, topk_indices = torch.topk(score, top_1, dim=-1)
        correct_tensor = topk_indices.eq(
            labels.unsqueeze(1).expand_as(topk_indices)
        )  # tensor of boolean values
        top1_correct = correct_tensor.cpu().sum().item()

        # correct
        predicted_labels = torch.argmax(score, dim=-1)
        correct_predictions = (predicted_labels == labels).cpu().sum().item()
        correct = correct_predictions

        metrics =  {
            "correct": correct,
            "top_10_correct": top10_correct,
            "top_5_correct": top5_correct,
            "top_1_correct": top1_correct,
        }
        batch_size = logits.shape[0]
        for k, v in metrics.items():
            metrics[k] = v / batch_size if batch_size > 0 else 0
            
        return metrics
