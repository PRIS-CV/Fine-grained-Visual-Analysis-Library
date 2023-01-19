import numpy as np

from .lr_schedule import LRSchedule
from . import lr_schedule


class CosineAnnealSchedule(LRSchedule):
    
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)
        self.update_level = 'epoch'

    def step(self, current_epoch, total_epoch, **kwargs):
        cos_inner = np.pi * (current_epoch % (total_epoch)) 
        cos_inner /= (total_epoch)
        cos_out = np.cos(cos_inner) + 1
        
        for pg in self.optimizer.param_groups:
            current_lr = pg['lr']
            pg['lr'] = float(current_lr / 2 * cos_out)

@lr_schedule("cosine_anneal_schedule")
def cosine_anneal_schedule(optimizer, batch_num_per_epoch, cfg:dict):
    return CosineAnnealSchedule(optimizer, **cfg)
