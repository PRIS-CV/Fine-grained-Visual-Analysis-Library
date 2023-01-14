__all__ = ["get_dataset"]

import os
import importlib

from fgvclib.datasets.datasets import FGVCDataset, available_datasets


__DATASET_DICT__ = {}


def get_dataset(name) -> FGVCDataset:
    r"""Return the dataset with the given name.
        Args: 
            name (str): 
                The name of dataset.
        Return: 
            (FGVCDataset): The dataset contructor method.
    """
    
    return __DATASET_DICT__[name]

def dataset(name):
    
    def register_function_fn(cls):
        if name in __DATASET_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, FGVCDataset):
            raise ValueError("Class %s is not a subclass of %s" % (cls, FGVCDataset))
        __DATASET_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.datasets.' + module_name)    
