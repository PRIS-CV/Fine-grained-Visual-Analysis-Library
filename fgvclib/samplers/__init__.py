import os
import importlib
from torch.utils.data.sampler import *


__SAMPLER_DICT__ = {
    "BatchSampler": BatchSampler,
    "RandomSampler": RandomSampler,
    "SequentialSampler": SequentialSampler,
    "SubsetRandomSampler": SubsetRandomSampler,
    "WeightedRandomSampler": WeightedRandomSampler,
}


def get_sampler(name) -> Sampler:
    r"""Return the dataset with the given name.
        Args: 
            name (str): 
                The name of sampler.
        Return: 
            (Sampler): The sampler contructor method.
    """
    
    return __SAMPLER_DICT__[name]


def sampler(name):
    
    def register_function_fn(cls):
        if name in __SAMPLER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Sampler):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Sampler))
        __SAMPLER_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.samplers.' + module_name)    
