import os
import importlib


__UPDATE_ST_DICT__ = {}


def get_update_strategy(name):
    r"""Return the update strategy with the given name.
        Args: 
            name (str): 
                The name of update strategy.
        Return: 
            (function): The update strategy function.
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