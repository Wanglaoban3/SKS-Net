import torch.nn.functional as F
import torch.nn as nn


class MultiScaleLoss(nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.scale = scale

    def forward(self, predict, target):
        if isinstance(predict, tuple):
            loss = F.cross_entropy(predict[0], target, weight=self.weight) * self.scale[0]
            for i in range(1, len(self.scale)):
                loss += F.mse_loss(predict[i], target.float()) * self.scale[i]
        else:
            loss = F.cross_entropy(predict, target, weight=self.weight)
        return loss
