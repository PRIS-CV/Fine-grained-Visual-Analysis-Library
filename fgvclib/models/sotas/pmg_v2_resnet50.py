import torch.nn as nn

from ..backbones import resnet50_bc
from ..necks.multi_scale_conv import MultiScaleConv
from ..encoders import GlobalMaxPooling
from ..heads.classifier_2fc import Classifier_2FC

class PMG_V2_ResNet50(nn.Module):
    def __init__(self, backbone, encoding, necks, heads, criterions):
        super(PMG_V2_ResNet50, self).__init__()

        self.backbone = backbone
        self.necks = necks
        self.encoding = encoding
        self.heads = heads
        self.outputs_num = 3
        

    def cuda(self, device=None):
        self.backbone.cuda(device)
        self.necks.cuda(device)
        self.heads.cuda(device)
    
    def forward(self, x, block=[0, 0, 0, 0]):
        f1, f2, f3, f4, f5, f3_, f4_, f5_ = self.backbone(x, block)
        x = tuple([f3, f4, f5])
        f = tuple([f3_, f4_, f5_])
        x = self.necks(x)
        x = self.encoding(x)
        x = self.heads(x)
        
        if self.training:
            return x, f
        else:
            return sum(x) / len(x)