import torch.nn as nn
from yacs.config import CfgNode


def cross_entropy_loss(cfg: CfgNode) -> nn.Module:
    r"""Build the cross entropy loss function.
        Args:
            cfg (CfgNode): The root node of config.
        
        Return:
            nn.Module: The loss function.
    """

    return nn.CrossEntropyLoss()

def binary_cross_entropy_loss(cfg: CfgNode) -> nn.Module:
    r"""Build the binary cross entropy loss function.
        Args:
            cfg (CfgNode): The root node of config.
        
        Return:
            nn.Module: The loss function.
    """
    
    return nn.BCELoss()

def mean_square_error_loss(cfg: CfgNode) -> nn.Module:
    r"""Build the mean square error loss function.
        Args:
            cfg (CfgNode): The root node of config.
        
        Return:
            nn.Module: The loss function.
    """

    return nn.MSELoss()

