import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        BEC_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BEC_EXP) ** self.gamma * BCE

        return focal_loss