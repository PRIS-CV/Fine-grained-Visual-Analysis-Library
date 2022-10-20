from .pmg import PMG_ResNet50
from .pmg_v2 import PMG_V2_ResNet50
from .resnet50 import Baseline_ResNet50
from .mcl import MCL

__all__ = [
    'PMG_ResNet50', 'PMG_V2_ResNet50', 'Baseline_ResNet50', 'MCL'
]


def get_model(model_name):
    r"""Return the FGVC model with the given name.

        Args: 
            model_name (str): 
                The name of model.
        
        Return: 
            The model contructor method.
    """
    
    if model_name not in globals():
        raise NotImplementedError(f"Model {model_name} not found!\nAvailable models: {__all__}")
    return globals()[model_name]

