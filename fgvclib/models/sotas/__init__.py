from .pmg import PMG
from .pmg_v2 import PMG_V2
from .resnet50 import ResNet50, ResNet50_CutMix, ResNet50_MixUp
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

