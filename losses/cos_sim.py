import torch


def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Compute the cosine similarity loss. Input and target of shape [B, T, C].
    """
    pred = pred.reshape(pred.size(0), -1)  # [B, T * C]
    target = target.reshape(target.size(0), -1)  # [B, T * C]

    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 1 - cosine_similarity(pred, target)  # [B]

    return loss.mean()
