import torch
from torch import nn


class MMDLoss(nn.Module):
    def __init__(
        self,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: int = None,
    ):
        """
        Args:
            kernel_type: string, kernel function type, "linear" | "rbf"
            kernel_mul: float, kernel function parameter
            kernel_num: int, number of kernel to sum
            fix_sigma: int, fix sigma value for rbf kernel. Estimated from data if None
        """
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ):
        """Source and target of shape [M, D] where M = B * S"""

        M_2, D = source.size(0) + target.size(0), source.size(1)
        total = torch.cat([source, target], dim=0)  # [2M, D]

        total0 = total.unsqueeze(0).expand(M_2, M_2, D)  # [2M, 2M, D]
        total1 = total.unsqueeze(1).expand(M_2, M_2, D)  # [2M, 2M, D]

        # Sum of pairwise distance of D
        L2_distance = ((total0 - total1) ** 2).sum(2)  # [2M, 2M]

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            # Avg pairwise distance of distinct pairs, excluding self distance
            bandwidth = torch.sum(L2_distance.data) / (M_2**2 - M_2)  # [1]

        # Center middle scale
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)

        # Capture different scales with power of self.kernel_mul
        bandwidth_list = [
            bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
        ]
        # exp(||x - y||^2 / bandwidth^2)
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]  # K * [2M, 2M]
        return sum(kernel_val)  # [2M, 2M]

    def linear_mmd2(self, f_of_X: torch.Tensor, f_of_Y: torch.Tensor):
        """Source and target of shape [B, D]"""
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)  # [D]
        loss = delta.dot(delta.T)  # [D] -> [1]

        return loss

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        """Source and target of shape [B, T, D]"""

        B, T, D = source.size()

        # randomly select 10 samples from T
        samples = 2
        indices = torch.randperm(source.size(1))  # [3000]
        indices = indices[:samples]  # [S]

        # [B, S, D]
        source, target = source[:, indices, :], target[:, indices, :]

        # [B, S, D] -> [B*S, D]
        source, target = source.reshape(B * samples, D), target.reshape(B * samples, D)

        if self.kernel_type == "linear":
            return self.linear_mmd2(source, target)

        elif self.kernel_type == "rbf":

            M = B * samples

            # [2M, 2M]
            kernels = self.guassian_kernel(source, target)

            XX = torch.mean(kernels[:M, :M])
            YY = torch.mean(kernels[M:, M:])
            XY = torch.mean(kernels[:M, M:])
            YX = torch.mean(kernels[M:, :M])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
