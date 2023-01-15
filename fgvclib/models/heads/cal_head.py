# Copyright (c) PRIS-CV. All rights reserved.
import torch
import torch.nn as nn

from fgvclib.models.heads import head


class Classifier_1FC_WO_BN(nn.Module):
    r"""Classifier with one fully connected layer.

        Args: 
            in_dim (List[int]):
                Input dimension, the number of classifiers is decided by the length of in_dim list.
            class_num (int):
                Output dimension.
    """

    def __init__(self, in_dim:list, class_num:int):
        super(Classifier_1FC_WO_BN, self).__init__()
        self.class_num = class_num
        self.classifier_num = len(in_dim)
        self.classifiers = nn.ModuleList()
        for s in range(self.classifier_num):
            self.classifiers.append(nn.Sequential(
            nn.Linear(in_dim[s], class_num),
        ))
    
    def forward(self, inputs):
        
        outputs = [self.classifiers[s](inputs[s]) for s in range(self.classifier_num)]
        
        if len(outputs) > 1:
            return tuple(outputs)
        else:
            return outputs[0]

    def get_class_num(self):
        return self.class_num


@head("cal_head")
def classifier_1fc(cfg: dict, class_num: int) -> Classifier_1FC_WO_BN:
    
    assert 'in_dim' in cfg.keys()
    assert isinstance(cfg['in_dim'], list)
    return Classifier_1FC_WO_BN(in_dim=cfg['in_dim'], class_num=class_num)
            