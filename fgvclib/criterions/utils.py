from dataclasses import dataclass
import typing as t
from torch import Tensor

@dataclass
class LossItem:
    r"""A dataclass object for store training loss
        Args:
            name (string): The loss item name.
            value (torch.Tensor): The value of loss.
            weight (float, optional): The weight of current loss item, default is 1.0. 
    """

    name: str = ""
    value: Tensor = None
    weight: t.Optional[float] = 1.0

def compute_loss_value(items: t.List[LossItem]) -> Tensor:
    r"""A dataclass object for store training loss
        Args:
            items (List[LossItem]): The loss items.
        
        Return:
            Tensor: The total loss value.
    """

    total = 0.
    for item in items:
        total = total + item.weight * item.value
    return total

def detach_loss_value(items: t.List[LossItem]) -> t.Dict:
    r"""Detach loss value from GPU.
        Args:
            items (List[LossItem]): The loss items.
        
        Return:
            Dict: A loss information dict whose key is loss name, value is loss value.
    """
    
    loss_dict = {}
    for item in items:
        loss_dict.update({item.name: round(item.value.item(), 2)})
    return loss_dict
