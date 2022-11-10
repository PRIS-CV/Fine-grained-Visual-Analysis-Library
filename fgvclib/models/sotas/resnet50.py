import torch.nn as nn

from fgvclib.criterions.utils import LossItem
from fgvclib.transforms import CutMix, MixUp


class ResNet50(nn.Module):
    
    def __init__(self, backbone, necks=None, encoding=None, heads=None, criterions=None):
        super(ResNet50, self).__init__()

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


class ResNet50_MixUp(ResNet50):

    def __init__(self, backbone, necks=None, encoding=None, heads=None, criterions=None):
        super(ResNet50_MixUp, self).__init__(backbone, necks, encoding, heads, criterions)
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


class ResNet50_CutMix(ResNet50):

    def __init__(self, backbone, necks=None, encoding=None, heads=None, criterions=None):
        super(ResNet50_CutMix, self).__init__(backbone, necks, encoding, heads, criterions)
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
    