# Copyright (c) PRIS-CV. All rights reserved.
import torch.nn as nn

from fgvclib.models.heads import head


class Classifier_2FC(nn.Module):
    r"""Classifier with one fully connected layer.

        Args: 
            in_dim (List[int]):
                Input dimension, the number of classifiers is decided by the length of in_dim list.
            hid_dim (List[int]):
                Hidden dimension, should has same length with the input dimension list.
            class_num (int):
                Output dimension.
    """

    def __init__(self, in_dim, hid_dim, class_num):
        super(Classifier_2FC, self).__init__()
        self.class_num = class_num
        self.classifier_num = len(in_dim)
        self.classifiers = nn.ModuleList()
        for s in range(self.classifier_num):
            self.classifiers.append(nn.Sequential(
            nn.BatchNorm1d(in_dim[s]),
            nn.Linear(in_dim[s], hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, class_num),
        ))
    
    def forward(self, inputs):
        
        outputs = [self.classifiers[s](inputs[s]) for s in range(self.classifier_num)]
        
        if len(outputs) > 1:
            return tuple(outputs)
        else:
            return outputs[0]

    def get_class_num(self):
        return self.class_num

@head("classifier_2fc")
def classifier_2fc(cfg: dict, class_num: int) -> Classifier_2FC:
    assert 'in_dim' in cfg.keys()
    assert isinstance(cfg['in_dim'], list)
    assert 'hid_dim' in cfg.keys()
    assert isinstance(cfg['hid_dim'], int)
    
    return Classifier_2FC(in_dim=cfg['in_dim'], hid_dim=cfg['hid_dim'], class_num=class_num)  