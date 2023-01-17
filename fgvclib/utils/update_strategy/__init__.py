import os
import importlib

from fgvclib.utils.lr_schedules.lr_schedule import LRSchedule
from torch.optim import SGD

__UPDATE_ST_DICT__ = {}


def get_update_strategy(name) -> LRSchedule:
    r"""Return the dataset with the given name.
        Args: 
            name (str): 
                The name of dataset.
        Return: 
            (FGVCDataset): The dataset contructor method.
    """
    
    return __UPDATE_ST_DICT__[name]

def update_strategy(name):
    
    def register_function_fn(cls):
        if name in __UPDATE_ST_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __UPDATE_ST_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.utils.update_strategy.' + module_name)    