import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.mse(predictions, targets)
