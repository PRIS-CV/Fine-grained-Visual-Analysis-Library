from .pmg_resnet50 import PMG_ResNet50
from .pmg_v2_resnet50 import PMG_V2_ResNet50
from .baseline_resnet50 import Baseline_ResNet50
from .mcl import MCL

__all__ = [
    'PMG_ResNet50', 'PMG_V2_ResNet50', 'Baseline_ResNet50', 'MCL'
]


def get_model(model_name):
    """Return the model class with the given name."""
    if model_name not in globals():
        raise NotImplementedError(f"Model {model_name} not found!\nAvailable models: {__all__}")
    return globals()[model_name]

