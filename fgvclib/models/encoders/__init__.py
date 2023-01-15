__all__ = ["get_encoder"]


import importlib
import os


__ENCODER_DICT__ = {}


def get_encoder(name):
    return __ENCODER_DICT__[name]

def encoder(name):
    
    def register_function_fn(cls):
        if name in __ENCODER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __ENCODER_DICT__[name] = cls
        return cls
    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.models.encoders.' + module_name)
