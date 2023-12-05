import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from .lr_schedule import LRSchedule
from . import lr_schedule


class WarmUpCosineLRScheduler(LRSchedule):
    
    def __init__(self, optimizer, warmup_epochs, decay_epochs, min_lr, warmup_lr) -> None:
        super().__init__(optimizer)
        self.update_level = 'batch'
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.optimizer = optimizer

    def step(self, batch_idx, current_epoch, total_batch, **kwargs):
        num_steps = int(total_batch * total_batch)
        warmup_steps = int(self.warmup_epochs * total_batch)
        decay_steps = int(self.decay_epochs * total_batch)    
        lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=num_steps,
            # t_mul=1.,
            lr_min=self.min_lr,
            warmup_lr_init=self.min_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )           
        lr_scheduler.step_update(current_epoch * total_batch + batch_idx)
      

@lr_schedule("cosine_warmup_schedule")
def cosine_warmup_schedule(optimizer, batch_num_per_epoch, cfg:dict):
    return WarmUpCosineLRScheduler(optimizer, **cfg)
