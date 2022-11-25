__all__ = ["get_backbone"]


import importlib
import os


__BACKBONE_DICT__ = {}


def get_backbone(name):
    return __BACKBONE_DICT__[name]

def backbone(name):
    
    def register_function_fn(cls):
        if name in __BACKBONE_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __BACKBONE_DICT__[name] = cls
        return cls
    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.models.backbones.' + module_name)
