from .base_loss import cross_entropy_loss, binary_cross_entropy_loss
from .mutual_channel_loss import mutual_channel_loss
from .utils import LossItem, compute_loss_value, detach_loss_value

__all__ = ['cross_entropy_loss', 'binary_cross_entropy_loss', 'mutual_channel_loss']

def get_criterion(criterion_name):
    """Return the criterion with the given name."""
    if criterion_name not in globals():
        raise NotImplementedError(f"Criterion {criterion_name} not found!\nAvailable criterions: {__all__}")
    return globals()[criterion_name]
