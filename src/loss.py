import torch
import torch.nn as nn

class trainLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, output, labels):
        # output: (batch_size, labels_num)
        return self.criterion(output, labels)
