
from torch import nn

from .sota import FGVCSOTA
from fgvclib.criterions import LossItem

class MCL(FGVCSOTA):
    r"""
        Code of "The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification".
        Link: https://github.com/PRIS-CV/Mutual-Channel-Loss
    """

    def __init__(self, backbone: nn.Module, encoding: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(backbone, encoding, necks, heads, criterions)

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
