__all__ = ["get_neck"]


import importlib
import os


__NECK_DICT__ = {}


def get_neck(name):
    return __NECK_DICT__[name]

def neck(name):
    
    def register_function_fn(cls):
        if name in __NECK_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __NECK_DICT__[name] = cls
        return cls
    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.models.necks.' + module_name)
