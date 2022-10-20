import torch.nn as nn
from fgvclib.criterions import LossItem

class PMG_V2(nn.Module):

    BLOCKS = [[8, 8, 0, 0], [4, 4, 4, 0], [2, 2, 2, 2]]
    alpha = [0.01, 0.05, 0.1]
    
    def __init__(self, backbone:nn.Module, encoding:nn.Module, necks:nn.Module, heads:nn.Module, criterions:nn.Module):
        super(PMG_V2, self).__init__()

        self.backbone = backbone
        self.necks = necks
        self.encoding = encoding
        self.heads = heads
        self.criterions = criterions
        self.outputs_num = 3
    
    def forward(self, x, targets, step:int):
        f1, f2, f3, f4, f5, f3_, f4_, f5_ = self.backbone(x, self.BLOCKS[step])
        x = tuple([f3, f4, f5])
        f = tuple([f3_, f4_, f5_])
        x = self.necks(x)
        x = self.encoding(x)
        x = self.heads(x)

        outputs, features = x[step], f[step]
        # refine
        if self.training:
            losses = list()
            losses.append(LossItem(name=f'step {step}', value=self.criterions['cross_entropy_loss']['fn'](outputs, targets)))
            losses.append(LossItem(name=f'step {step}', value=self.criterions['mean_square_error_loss']['fn'](outputs, features), weight=self.alpha[step]))
            return x, losses
        else:
            return sum(x) / len(x)
