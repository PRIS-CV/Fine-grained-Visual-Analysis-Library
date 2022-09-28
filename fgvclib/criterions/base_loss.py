import torch.nn as nn

def cross_entropy_loss(cfg):
    return nn.CrossEntropyLoss()

def binary_cross_entropy_loss(cfg):
    return nn.BCELoss()

def mean_square_error_loss(cfg):
    return nn.MSELoss()

