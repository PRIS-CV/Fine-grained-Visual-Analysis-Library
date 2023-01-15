from typing import Iterable
import torch.nn as nn
from torch.optim import Optimizer


from fgvclib.utils.update_strategy import get_update_strategy
from fgvclib.utils.logger import Logger
from fgvclib.utils.lr_schedules import LRSchedule

def update_model(
    model: nn.Module, optimizer: Optimizer, pbar:Iterable, lr_schedule:LRSchedule=None,
    strategy:str="general_updating", use_cuda:bool=True, logger:Logger=None, 
    epoch:int=None, total_epoch:int=None, **kwargs
):
    r"""Update the FGVC model and record losses.

    Args:
        model (nn.Module): The FGVC model.
        optimizer (Optimizer): The Logger object.
        pbar (Iterable): The iterable object provide training data.
        lr_schedule (LRSchedule): The lr schedule updating class.
        strategy (string): The update strategy.
        use_cuda (boolean): Whether to use GPU to train the model.
        logger (Logger): The Logger object.
        epoch (int): The current epoch number.
        total_epoch (int): The total epoch number.
    """
    model.train()
    mean_loss = 0.
    for batch_idx, train_data in enumerate(pbar):
        losses_info = get_update_strategy(strategy)(model, train_data, optimizer, use_cuda)
        mean_loss = (mean_loss * batch_idx + losses_info['iter_loss']) / (batch_idx + 1)
        losses_info.update({"mean_loss": mean_loss})
        logger(losses_info, step=batch_idx)
        pbar.set_postfix(losses_info)
        if lr_schedule.update_level == 'batch_update':
            lr_schedule.step(optimizer=optimizer, batch_idx=batch_idx, batch_size=len(train_data), current_epoch=epoch, total_epoch=total_epoch)
    
    if lr_schedule.update_level == 'epoch_update':
        lr_schedule.step(optimizer=optimizer, current_epoch=epoch, total_epoch=total_epoch)



    