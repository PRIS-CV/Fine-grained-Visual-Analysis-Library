__all__ = ["get_metric"]

import os
import importlib


__METRIC_DICT__ = {}


def get_metric(name):
    r"""Return the metric with the given name.

        Args: 
            name (str): 
                The name of metric.
        
        Return: 
            The metric contructor method.
    """

    return __METRIC_DICT__[name]

def get_metric(name):
    return __METRIC_DICT__[name]

def metric(name):
    
    def register_function_fn(cls):
        if name in __METRIC_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __METRIC_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.metrics.' + module_name)  
