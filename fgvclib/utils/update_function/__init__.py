import os
import importlib


__UPDATE_FN_DICT__ = {}


def get_update_function(name):
    r"""Return the update model function with the given name.
        Args: 
            name (str): 
                The name of update function.
        Return: 
            (function): The update function.
    """
    
    return __UPDATE_FN_DICT__[name]

def update_function(name):
    
    def register_function_fn(cls):
        if name in __UPDATE_FN_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __UPDATE_FN_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.utils.update_function.' + module_name)    