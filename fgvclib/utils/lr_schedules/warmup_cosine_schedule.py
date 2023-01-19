from torch.optim.lr_scheduler import LambdaLR

from . import lr_schedule



class WarmupLinearSchedule(LambdaLR):
    

    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        self.update_level = "batch"

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.total_steps - step) / float(max(1.0, self.total_steps - self.warmup_steps)))
    
    def step(self, **kwargs):
        return super().step()


@lr_schedule("warmup_linear_schedule")
def warmup_linear_schedule(optimizer, batch_num_per_epoch, cfg:dict):
    return WarmupLinearSchedule(optimizer=optimizer, **cfg)

