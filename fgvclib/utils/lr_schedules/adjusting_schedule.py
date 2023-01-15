from .lr_schedule import LRSchedule
from . import lr_schedule

@lr_schedule("adjusting_schedule")
class Adjusting_Schedule(LRSchedule):
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.base_rate = cfg["base_rate"]
        self.base_duration = cfg["base_duration"]
        self.base_lr = cfg["base_lr"]
        self.update_level = 'batch_update'

    def step(self, optimizer, batch_idx, current_epoch, batch_size, **kwargs):
        iter = float(batch_idx) / batch_size
        lr = self.base_lr * pow(self.base_rate, (current_epoch + iter) / self.base_duration)
        for pg in optimizer.param_groups:
           pg['lr'] = lr
