import torch.nn as nn

from fgvclib.criterions.utils import LossItem

class Baseline_ResNet50(nn.Module):
    
    def __init__(self, backbone, necks=None, encoding=None, heads=None, criterions=None):
        super(Baseline_ResNet50, self).__init__()

        self.backbone = backbone
        self.necks = necks
        self.encoding = encoding
        self.heads = heads
        self.criterions = criterions
    
    def forward(self, x, targets=None):
        x = self.infer(x)
        if self.training:
            losses = list()
            losses.append(LossItem(name='cross_entropy_loss', value=self.criterions['cross_entropy_loss']['fn'](x, targets)))
            return x, losses
        
        return x

    def infer(self, x):
        f1, f2, f3, f4, f5 = self.backbone(x)
        x = tuple([f5])
        x = self.encoding(x)
        x = self.heads(x)
        return x
        