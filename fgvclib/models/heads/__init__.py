__all__ = ["get_head"]


import importlib
import os


__HEAD_DICT__ = {}


def get_head(name):
    return __HEAD_DICT__[name]

def head(name):
    
    def register_function_fn(cls):
        if name in __HEAD_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __HEAD_DICT__[name] = cls
        return cls
    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.models.heads.' + module_name)
        