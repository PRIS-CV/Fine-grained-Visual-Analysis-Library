import math
from torch.optim.lr_scheduler import LambdaLR
from . import lr_schedule
from timm.scheduler.cosine_lr import CosineLRScheduler
from .lr_schedule import LRSchedule


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        self.update_level = "batch"

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

    def step(self, **kwargs):
        return super().step()


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
            lr_min=self.min_lr,
            warmup_lr_init=self.min_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
        lr_scheduler.step_update(current_epoch * total_batch + batch_idx)


@lr_schedule("warmup_cosine_schedule")
def warmup_cosine_schedule(optimizer, batch_num_per_epoch, cfg: dict):
    return WarmupCosineSchedule(optimizer=optimizer, **cfg)


@lr_schedule("warmup_cosine_schedule_timm")
def cosine_warmup_schedule(optimizer, batch_num_per_epoch, cfg: dict):
    return WarmUpCosineLRScheduler(optimizer, **cfg)
