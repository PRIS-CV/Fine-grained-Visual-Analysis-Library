import torch
from . import criterion

class SoftTargetCrossEntropy(torch.nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * x, dim=-1)
        return loss.mean()

@criterion("soft_target_cross_entropy_loss")
def soft_target_cross_entropy_loss(cfg=None):
    return SoftTargetCrossEntropy()        