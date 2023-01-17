import torch.nn as nn
from yacs.config import CfgNode

from fgvclib.models.sotas.sota import FGVCSOTA
from fgvclib.criterions import LossItem
from fgvclib.models.sotas import fgvcmodel


@fgvcmodel("PMG_V2")
class PMG_V2(FGVCSOTA):
    r"""
        Code of "Progressive Learning of Category-Consistent Multi-Granularity Features for Fine-Grained Visual Classification". 
        It is the extended version of "Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches" (ECCV2020).
        Link: https://github.com/RuoyiDu/PMG-V2
    """

    BLOCKS = [[8, 8, 0, 0], [4, 4, 4, 0], [2, 2, 2, 2]]
    alpha = [0.01, 0.05, 0.1]
    
    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
        self.BLOCKS = self.args["BLOCKS"]
        self.outputs_num = self.args["outputs_num"]
        self.alpha = self.args["alpha"]

    def forward(self, x, targets=None, step:int=None, batch_size:int=None):
        
        if step is not None:
            block = self.BLOCKS[step]
        else:
            block = [0, 0, 0, 0]
        
        x, f = self.infer(x, block)

        if self.training:
            outputs, features = x[step], f[step]
            losses = list()
            losses.append(LossItem(name=f'step {step}', value=self.criterions['cross_entropy_loss']['fn'](outputs, targets)))
            losses.append(LossItem(name=f'step {step}', value=self.criterions['mean_square_error_loss']['fn'](features[: batch_size], features[batch_size:]), weight=self.alpha[step]))
            return x, losses
        else:
            return sum(x) / len(x)

    def infer(self, x, block):
        
        _, _, f3, f4, f5, f3_, f4_, f5_ = self.backbone(x, block)
        x = tuple([f3, f4, f5])
        f = tuple([f3_, f4_, f5_])
        x = self.necks(x)
        x = self.encoder(x)
        x = self.heads(x)

        return x, f
        

            
