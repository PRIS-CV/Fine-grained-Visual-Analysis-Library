from .lr_schedule import LRSchedule
from . import lr_schedule


class Adjusting_Schedule(LRSchedule):
    
    def __init__(self, optimizer, base_rate, base_duration, base_lr) -> None:
        super().__init__(optimizer)
        
        self.base_rate = base_rate
        self.base_duration = base_duration
        self.base_lr = base_lr
        self.update_level = "batch"

    def step(self, batch_idx, current_epoch, total_batch, **kwargs):
        iter = float(batch_idx) / total_batch
        lr = self.base_lr * pow(self.base_rate, (current_epoch + iter) / self.base_duration)
        for pg in self.optimizer.param_groups:
           pg['lr'] = lr


@lr_schedule("adjusting_schedule")
def adjusting_schedule(optimizer, batch_num_per_epoch, cfg:dict):
    return Adjusting_Schedule(optimizer, **cfg)
