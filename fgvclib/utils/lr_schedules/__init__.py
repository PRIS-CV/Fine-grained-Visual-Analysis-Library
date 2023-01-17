import os
import importlib
from torch.optim.lr_scheduler import _LRScheduler

from fgvclib.utils.lr_schedules.lr_schedule import LRSchedule


__LR_SCHEDULE_DICT__ = {}


def get_lr_schedule(name) -> LRSchedule:
    r"""Return the dataset with the given name.
        Args: 
            name (str): 
                The name of dataset.
        Return: 
            (FGVCDataset): The dataset contructor method.
    """
    
    return __LR_SCHEDULE_DICT__[name]

def lr_schedule(name):
    
    def register_function_fn(cls):
        if name in __LR_SCHEDULE_DICT__:
            raise ValueError("Name %s already registered!" % name)
        # if not issubclass(cls, LRSchedule) and not issubclass(cls, _LRScheduler):
        #     raise ValueError("Class %s is not a subclass of %s or %s" % (cls, LRSchedule, _LRScheduler))
        __LR_SCHEDULE_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.utils.lr_schedules.' + module_name)    

