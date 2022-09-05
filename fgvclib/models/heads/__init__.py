from .classifier_1fc import classifier_1fc
from .classifier_2fc import classifier_2fc

__all__ = ['classifier_1fc', 'classifier_2fc']

def get_head(head_name):
    """Return the backbone with the given name."""
    if head_name not in globals():
        raise NotImplementedError(f"Head not found: {head_name}\nAvailable heads: {__all__}")
    return globals()[head_name]