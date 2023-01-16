from torch.optim.lr_scheduler import LambdaLR

from .lr_schedule import LRSchedule


class ConstantSchedule(LambdaLR, LRSchedule):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, total_epoch=-1):
        super(ConstantSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=total_epoch)