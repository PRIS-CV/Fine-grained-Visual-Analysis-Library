import torch.nn as nn
import random

from .sota import FGVCSOTA
from fgvclib.criterions.utils import LossItem

class PMG(FGVCSOTA):
    r"""
        Code of Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV2020).
        Link: https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
    """
    
    
    def __init__(self, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(backbone, encoder, necks, heads, criterions)
        
        self.outputs_num = 4
    
    def infer(self, x):
        _, _, f3, f4, f5 = self.backbone(x)
        x = tuple([f3, f4, f5])
        x = self.necks(x)
        x = self.encoding(x, concat=True)
        x = self.heads(x)
        return x
    
    def forward(self, x, targets=None, step=None):
        if self.training:
            losses = list()
            outputs = self.jigsaw_generator(x, 2 ** (self.outputs_num - (step + 1)))
            step_out = self.infer(outputs)[step]
            losses.append(LossItem(name=f'step {step}', value=self.criterions['cross_entropy_loss']['fn'](step_out, targets)))
            return outputs, losses
        else:
            outputs = self.infer(x)
            return sum(outputs) / len(outputs)
    
    def jigsaw_generator(self, images, n):
        l = []
        for a in range(n):
            for b in range(n):
                l.append([a, b])
        block_size = 448 // n
        rounds = n ** 2
        random.shuffle(l)
        jigsaws = images.clone()
        for i in range(rounds):
            x, y = l[i]
            temp = jigsaws[..., 0:block_size, 0:block_size].clone()
            jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                    y * block_size:(y + 1) * block_size].clone()
            jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

        return jigsaws
    