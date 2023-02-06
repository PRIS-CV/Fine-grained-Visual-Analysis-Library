import os
import importlib

__EVAL_FN_DICT__ = {}


def get_evaluate_function(name):
    r"""Return the evaluate function with the given name.
        Args: 
            name (str): 
                The name of evaluate function.
        Return: 
            (function): The evaluate function.
    """
    
    return __EVAL_FN_DICT__[name]

def evaluate_function(name):
    
    def register_function_fn(cls):
        if name in __EVAL_FN_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __EVAL_FN_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.utils.evaluate_function.' + module_name)
   