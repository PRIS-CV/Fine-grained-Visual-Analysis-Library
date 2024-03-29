
from torch import nn
from yacs.config import CfgNode

from fgvclib.models.sotas.sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel
from fgvclib.criterions import LossItem

@fgvcmodel("MCL")
class MCL(FGVCSOTA):
    r"""
        Code of "The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification".
        Link: https://github.com/PRIS-CV/Mutual-Channel-Loss
    """

    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
        self.num_classes = cfg.CLASS_NUM

    
    def forward(self, x, targets=None):
        x = self.backbone(x)
        if self.training:
            losses = list()
            losses.extend(self.criterions['mutual_channel_loss']['fn'](x, targets, self.num_classes))
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = tuple([x])
        x = self.heads(x)

        if self.training:
            losses.append(LossItem(name="cross_entropy_loss", 
                                   value=self.criterions['cross_entropy_loss']['fn'](x, targets), 
                                   weight=self.criterions['cross_entropy_loss']['w'])) 

            return x, losses

        return x
