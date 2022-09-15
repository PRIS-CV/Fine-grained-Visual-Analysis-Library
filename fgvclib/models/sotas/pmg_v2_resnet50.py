import torch.nn as nn

from ..backbones import resnet50_bc
from ..necks.multi_scale_conv import MultiScaleConv
from ..encoders import GlobalMaxPooling
from ..heads.classifier_2fc import Classifier_2FC

class PMG_V2_ResNet50(nn.Module):
    def __init__(self, pretrain=True, classes_num=200):
        super(PMG_V2_ResNet50, self).__init__()

        self.backbone = resnet50_bc(pretrained=pretrain)
        self.necks = MultiScaleConv(3, [512, 1024, 2048], [512, 512, 512], [1024, 1024, 1024])
        self.encoding = GlobalMaxPooling()
        self.heads = Classifier_2FC([1024, 1024, 1024], 512, classes_num)
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