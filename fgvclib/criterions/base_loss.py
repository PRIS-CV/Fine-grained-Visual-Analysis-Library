import torch.nn as nn
from yacs.config import CfgNode

from fgvclib.criterions import criterion

@criterion("cross_entropy_loss")
def cross_entropy_loss(cfg: CfgNode) -> nn.Module:
    r"""Build the cross entropy loss function.
        Args:
            cfg (CfgNode): The root node of config.
        
        Return:
            nn.Module: The loss function.
    """

    return nn.CrossEntropyLoss()

@criterion("binary_cross_entropy_loss")
def binary_cross_entropy_loss(cfg: CfgNode) -> nn.Module:
    r"""Build the binary cross entropy loss function.
        Args:
            cfg (CfgNode): The root node of config.
        
        Return:
            nn.Module: The loss function.
    """
    
    return nn.BCELoss()

@criterion("mean_square_error_loss")
def mean_square_error_loss(cfg: CfgNode) -> nn.Module:
    r"""Build the mean square error loss function.
        Args:
            cfg (CfgNode): The root node of config.
        
        Return:
            nn.Module: The loss function.
    """
    
    return nn.MSELoss()

