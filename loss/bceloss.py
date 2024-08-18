import torch.nn as nn
import torch


class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight # 二分类中正负样本的权重，第一项为负类权重，第二项为正类权重

    def forward(self, input, target):
        # input = torch.clamp(input, min=1e-8, max=1-1e-8)
        bce = - self.weight[1] * target * torch.log(input) - (1 - target) * self.weight[0] * torch.log(1 - input)
        return torch.mean(bce)
