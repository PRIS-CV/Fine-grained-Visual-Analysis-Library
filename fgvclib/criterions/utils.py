from dataclasses import dataclass
import typing as t
from torch import Tensor
from decimal import Decimal

@dataclass
class LossItem:
    name: str = ""
    value: t.Any = None
    weight: t.Optional[t.Union[float, Tensor]] = 1.0

def compute_loss_value(items: t.List[LossItem]) -> Tensor:
        """
        Should return the value that will be used later for backward propagation
        """
        total = 0.
        for item in items:
            total = total + item.weight * item.value
        return total

def detach_loss_value(items: t.List[LossItem]) -> t.Dict:
    loss_dict = {}
    for item in items:
        loss_dict.update({item.name: round(item.value.item(), 2)})
    return loss_dict
