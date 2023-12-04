import os
import importlib
from .mixup import MixUp
from .cutmix import CutMix
from .mixup_cutmix import MixUpCutMix


__all__ = ["get_transform"]


__TRANSFORM_DICT__ = {}


def get_transform(name):
    r"""Return the dataset with the given name.
        Args: 
            name (str): 
                The name of dataset.
        Return: 
            (FGVCDataset): The dataset contructor method.
    """
    
    return __TRANSFORM_DICT__[name]

def transform(name):
    
    def register_function_fn(cls):
        if name in __TRANSFORM_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __TRANSFORM_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.transforms.' + module_name)    