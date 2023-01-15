from .lr_schedule import LRSchedule
from . import lr_schedule

@lr_schedule("adjusting_schedule")
class Adjusting_Schedule(LRSchedule):
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.base_rate = 0.9
        self.base_duration = 2.0
        self.base_lr = 1e-3
        self.update_level = 'batch_update'

    def step(self, optimizer, batch_idx, current_epoch, batch_size, **kwargs):
        iter = float(batch_idx) / batch_size
        lr = self.base_lr * pow(self.base_rate, (current_epoch + iter) / self.base_duration)
        for pg in optimizer.param_groups:
           pg['lr'] = lr
