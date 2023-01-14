import os
import importlib
import typing as t

from .utils import *


__CRITERION_DICT__ = {}


def get_criterion(name) -> t.Callable:
    r"""Return the criterion with the given name.
        Args: 
            name (str): 
                The name of criterion.
        Return: 
            (FGVCDataset): The criterion contructor method.
    """
    
    return __CRITERION_DICT__[name]

def criterion(name):
    
    def register_function_fn(cls):
        if name in __CRITERION_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __CRITERION_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.criterions.' + module_name)   
