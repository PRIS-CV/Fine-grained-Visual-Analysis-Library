import torch.nn as nn

def cross_entropy_loss(cfg=None):
    return nn.CrossEntropyLoss()

def binary_cross_entropy_loss(cfg=None):
    return nn.BCELoss()

