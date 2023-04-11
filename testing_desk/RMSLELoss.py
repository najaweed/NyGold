import torch
import torch.nn as nn


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predict, target):
        return torch.sqrt(self.mse(torch.log(predict + 1), torch.log(target + 1)))
