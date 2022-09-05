from .progressive_updating_with_jigsaw import progressive_updating_with_jigsaw
from .progressive_updating_consistency_constraint import progressive_updating_consistency_constraint
from .general_updating import general_updating

__all__ = [
    'progressive_updating_with_jigsaw', 'progressive_updating_consistency_constraint', 'general_updating'
]

def get_update_strategy(strategy_name):
    """Return the update strategy with the given name."""
    if strategy_name not in globals():
        raise NotImplementedError(f"Strategy not found: {strategy_name}\nAvailable strategy: {__all__}")
    return globals()[strategy_name]
