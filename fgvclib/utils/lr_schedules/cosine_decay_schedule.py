import math
import numpy as np

from . import lr_schedule
from .lr_schedule import LRSchedule

class WarmupCosineDecaySchedule(LRSchedule):
    
    def __init__(self, optimizer, max_epochs, warmup_steps, max_lr, batch_num_per_epoch, decay_type) -> None:
        super().__init__(optimizer)

        total_batchs = max_epochs * batch_num_per_epoch
        iters = np.arange(total_batchs - warmup_steps)
        self.last_iter = -1
        self.update_level = 'batch'

        if decay_type == 1:
            schedule = np.array([1e-12 + 0.5 * (max_lr - 1e-12) * (1 + math.cos(math.pi * t / total_batchs)) for t in iters])
        elif decay_type == 2:
            schedule = max_lr * np.array([math.cos(7 * math.pi * t / (16 * total_batchs)) for t in iters])
        else:
            raise ValueError("Not support this decay type")

        if warmup_steps > 0:
            warmup_lr_schedule = np.linspace(1e-9, max_lr, warmup_steps)
            schedule = np.concatenate((warmup_lr_schedule, schedule))

        self.schedule = schedule
        self.step()
    
    def step(self):
        self.last_iter += 1
        for pg in self.optimizer.param_groups:
           pg['lr'] = self.schedule[self.last_iter]
        

@lr_schedule("warmup_cosine_decay_schedule")
def warmup_cosine_decay_schedule(optimizer, batch_num_per_epoch, cfg:dict):
    return WarmupCosineDecaySchedule(optimizer=optimizer, batch_num_per_epoch=batch_num_per_epoch, **cfg)
