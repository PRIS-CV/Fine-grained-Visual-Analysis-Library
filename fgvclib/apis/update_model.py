from typing import Iterable
import torch.nn as nn
from torch.optim import Optimizer

from fgvclib.utils.update_strategy import get_update_strategy
from fgvclib.utils.logger import Logger

def update_model(model: nn.Module, optimizer: Optimizer, pbar:Iterable, strategy:str="general_updating", use_cuda:bool=True, logger:Logger=None):
    r"""Update the FGVC model and record losses.

    Args:
        model (nn.Module): The FGVC model.
        optimizer (Optimizer): The Logger object.
        pbar (Iterable): A iterable object provide training data.
        strategy (string): The update strategy.
        use_cuda (boolean): Whether to use GPU to train the model.
        logger (Logger): The Logger object.
    """
    model.train()
    mean_loss = 0.
    for batch_idx, train_data in enumerate(pbar):
        losses_info = get_update_strategy(strategy)(model, train_data, optimizer, use_cuda)
        mean_loss = (mean_loss * batch_idx + losses_info['iter_loss']) / (batch_idx + 1)
        losses_info.update({"mean_loss": mean_loss})
        logger(losses_info, step=batch_idx)
        pbar.set_postfix(losses_info)
    