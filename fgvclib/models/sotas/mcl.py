import numpy as np
from torch import nn
from fgvclib.criterions import LossItem, compute_loss_value

class MCL(nn.Module):
    def __init__(self, backbone=None, necks=None, encoding=None, heads=None, criterions=None):

        super(MCL, self).__init__() 

        self.backbone = backbone
        self.necks = necks
        self.encoding = encoding
        self.heads = heads
        self.criterions = criterions

    def forward(self, x, targets):
        x = self.backbone(x)
        if self.training:
            losses = list()
            losses.extend(self.criterions['mutual_channel_loss']['fn'](x, targets, self.heads.get_class_num()))
        
        x = self.encoding(x)
        x = x.view(x.size(0), -1)
        x = tuple([x])
        x = self.heads(x)

        if self.training:
            losses.append(LossItem(name="cross_entropy_loss", 
                                   value=self.criterions['cross_entropy_loss']['fn'](x, targets), 
                                   weight=self.criterions['cross_entropy_loss']['w'])) 

            return x, losses

        return x
