import torch.nn as nn

from . import criterion

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

@criterion("center_loss")
def center_loss(cfg=None):
    return CenterLoss()
