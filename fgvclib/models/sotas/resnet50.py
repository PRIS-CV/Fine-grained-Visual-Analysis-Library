import torch.nn as nn
from yacs.config import CfgNode

from fgvclib.models.sotas.sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel
from fgvclib.criterions.utils import LossItem
from fgvclib.transforms import CutMix, MixUp


@fgvcmodel("ResNet50")
class ResNet50(FGVCSOTA):
    
    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
    
    def forward(self, x, targets=None):
        x = self.infer(x)
        if self.training:
            losses = list()
            losses.append(LossItem(name='cross_entropy_loss', value=self.criterions['cross_entropy_loss']['fn'](x, targets)))
            return x, losses
        
        return x

    # def forward(self, x, targets=None):
    #     x = self.infer(x)
    #     if self.training:
    #         return self.criterions['cross_entropy_loss']['fn'](x, targets)
        
    #     return x

    def infer(self, x):
        _, _, _, _, f5 = self.backbone(x)
        x = tuple([f5])
        x = self.encoder(x)
        x = self.heads(x)
        return x


@fgvcmodel("ResNet50_MixUp")
class ResNet50_MixUp(ResNet50):

    def __init__(self, backbone: nn.Module, encoding: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(backbone, encoding, necks, heads, criterions)
        self.mix_up = MixUp()
        
    def forward(self, x, targets=None):
        if self.training:
            x, target_a, target_b, lam = self.mix_up.aug_data(x, targets)
            x = self.infer(x)
            loss_value = self.mix_up.aug_criterion(self.criterions['cross_entropy_loss']['fn'], x, target_a, target_b, lam)
            losses = list()
            losses.append(LossItem(name='mixup_cross_entropy_loss', value=loss_value))
            return x, losses
        else:
            x = self.infer(x)

        return x


@fgvcmodel("ResNet50_CutMix")
class ResNet50_CutMix(ResNet50):

    def __init__(self, backbone: nn.Module, encoding: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(backbone, encoding, necks, heads, criterions)
        self.cut_mix = CutMix()
        
    def forward(self, x, targets=None):
        if self.training:
            x, target_a, target_b, lam = self.cut_mix.aug_data(x, targets)
            x = self.infer(x)
            loss_value = self.cut_mix.aug_criterion(self.criterions['cross_entropy_loss']['fn'], x, target_a, target_b, lam)
            losses = list()
            losses.append(LossItem(name='cutmix_cross_entropy_loss', value=loss_value))
            return x, losses
        else:
            x = self.infer(x)

        return x
    