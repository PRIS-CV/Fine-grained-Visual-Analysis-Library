import torch.nn as nn
import random

from ..backbones import resnet50
from ..necks.multi_scale_conv import MultiScaleConv
from ..encoding import GlobalMaxPooling
from ..heads.classifier_2fc import Classifier_2FC
from fgvclib.criterions.utils import LossItem

class PMG_ResNet50(nn.Module):
    def __init__(self, backbone=None, necks=None, encoding=None, heads=None, criterions=None):
        
        super(PMG_ResNet50, self).__init__()

        self.backbone = backbone
        self.necks = necks
        self.encoding = encoding
        self.heads = heads
        self.criterions = criterions
        self.outputs_num = 4
    
    def infer(self, x):
        f1, f2, f3, f4, f5 = self.backbone(x)
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