import gc
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.linear_x1 = torch.nn.LazyLinear(dim)
        self.linear_x2 = torch.nn.LazyLinear(dim)

    def forward(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        segment_level: bool = True,
    ) -> dict[str, float]:
        """
        Computes CLIP loss on two embeddings, x_1 and x_2. Both of shape [B, C, T]
        If segment level true, computes similarity between segments. Otherwise at the
        time step level. Metrics always calculated at segment level.

        Returns:
            clip_loss: torch.Tensor, shape [B]
            metrics: dict[str, float]
        """
        assert x_1.size() == x_2.size()
        B, C, T = x_1.size()

        # Normalize embeddings
        x_1 = self.linear_x1(x_1)
        x_2 = self.linear_x2(x_2)

        inv_norms = 1 / (1e-8 + x_1.norm(dim=(1, 2), p=2))  # [B]

        # Segment level
        # Compute similarity, [B, C, T] x [B, C, T] -> [B, B]
        segment_level_logits = (
            torch.einsum("bct,dct,d->bd", x_1, x_2, inv_norms) / self.temperature
        )
        segment_level_targets = torch.arange(
            x_1.size(0), device=x_1.device
        )  # Diagonal targets
        segment_level_probs = F.log_softmax(segment_level_logits, dim=-1)

        if segment_level:
            clip_loss = F.cross_entropy(
                segment_level_probs, segment_level_targets, reduction="mean"
            )
        # Time step level, optional
        else:
            # Shorten time steps for efficiency since clip scales quadratically
            keep_T = max(1, T // 5)
            x1_kept_list, x2_kept_list = [], []
            indices = torch.randperm(T, device=x_1.device)[:keep_T]

            for b in range(B):
                x1_kept_list.append(x_1[b, :, indices])
                x2_kept_list.append(x_2[b, :, indices])

            x_1 = torch.stack(x1_kept_list, dim=0)  # [B, C, T]
            x_2 = torch.stack(x2_kept_list, dim=0)

            del x1_kept_list, x2_kept_list
            gc.collect()
            torch.cuda.empty_cache()

            # [B, C, T] -> [B * T, C]
            x_1, x_2 = (x_1.reshape(B * keep_T, C), x_2.reshape(B * keep_T, C))

            x_1_norm = x_1 / (x_1.norm(dim=1, keepdim=True) + 1e-8)
            x_2_norm = x_2 / (x_2.norm(dim=1, keepdim=True) + 1e-8)

            logits = x_1_norm @ x_2_norm.transpose(0, 1)  # [B*T, B*T]
            logits = logits / self.temperature

            # Diagonal targets
            time_step_targets = torch.arange(B * keep_T, device=x_1.device)  # [B * T]
            time_step_probs = F.log_softmax(logits, dim=-1)  # [B * T]
            clip_loss = F.cross_entropy(
                time_step_probs, time_step_targets, reduction="mean"
            )

        return {
            "loss": clip_loss,  # Still on device
            "metrics": self.eval_metrics(
                segment_level_probs, segment_level_targets
            ),  # on CPU
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
