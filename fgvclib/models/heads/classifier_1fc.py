# Copyright (c) PRIS-CV. All rights reserved.
import torch
import torch.nn as nn

class Classifier_1FC(nn.Module):
    """Classifier with one fully connected layer.

    Note that ...

    Args:
        
    """

    def __init__(self, in_dim:list=[2048,], class_num:int=None):
        super(Classifier_1FC, self).__init__()

        self.classifier_num = len(in_dim)
        self.classifiers = nn.ModuleList()
        for s in range(self.classifier_num):
            self.classifiers.append(nn.Sequential(
            nn.BatchNorm1d(in_dim[s]),
            nn.Linear(in_dim[s], class_num),
        ))
    
    def forward(self, inputs):
        
        outputs = [self.classifiers[s](inputs[s]) for s in range(self.classifier_num)]
        
        if len(outputs) > 1:
            return tuple(outputs)
        else:
            return outputs[0]

def classifier_1fc(class_num: int, cfg: dict) -> Classifier_1FC:
    assert 'in_dim' in cfg.keys()
    assert isinstance(cfg['in_dim'], list)

    return Classifier_1FC(in_dim=cfg['in_dim'], class_num=class_num)
            