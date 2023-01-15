import numpy as np

from .lr_schedule import LRSchedule
from . import lr_schedule

@lr_schedule("cosine_anneal_schedule")
class CosineAnnealSchedule(LRSchedule):
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.update_level = 'epoch_update'

    def step(self, optimizer, current_epoch, total_epoch, **kwargs):
        cos_inner = np.pi * (current_epoch % (total_epoch)) 
        cos_inner /= (total_epoch)
        cos_out = np.cos(cos_inner) + 1
        
        for pg in optimizer.param_groups:
            current_lr = pg['lr']
            pg['lr'] = float(current_lr / 2 * cos_out)
