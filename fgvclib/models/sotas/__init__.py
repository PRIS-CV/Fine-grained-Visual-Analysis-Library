__all__ = ["get_model"]


import importlib
import os

from fgvclib.models.sotas.sota import FGVCSOTA

__MODEL_DICT__ = {}


def get_model(name):
    r"""Return the FGVC model with the given name.

        Args: 
            model_name (str): 
                The name of model.
        
        Return: 
            The model contructor method.
    """
    return __MODEL_DICT__[name]


def fgvcmodel(name):
    
    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, FGVCSOTA):
            raise ValueError("Class %s is not a subclass of %s" % (cls, FGVCSOTA))
        __MODEL_DICT__[name] = cls
        return cls
    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.models.sotas.' + module_name)
