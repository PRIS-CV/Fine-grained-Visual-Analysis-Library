from typing import Iterable
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch
import torch.functional as F

from . import update_function
from fgvclib.utils.logger import Logger
from fgvclib.criterions import compute_loss_value, detach_loss_value



@update_function("update_swin_transformer")
def update_swin_transformer(model: nn.Module, optimizer: Optimizer, pbar:Iterable, lr_schedule=None,
    strategy:str="update_swint", use_cuda:bool=True, logger:Logger=None, 
    epoch:int=None, total_epoch:int=None, amp:bool=False, use_selection=False, cfg=None, **kwargs,
):  
    scaler = GradScaler()

    optimizer.zero_grad()

    for batch_id, (inputs, targets) in enumerate(pbar):
        model.train()
        if lr_schedule.update_level == "batch":
            iteration = epoch * len(pbar) + batch_id
            lr_schedule.step(iteration)

        if lr_schedule.update_level == "epoch":
            lr_schedule.step()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        with autocast():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            out, losses = model(inputs, targets)
            total_loss = compute_loss_value(losses)
            if hasattr(model, 'update_freq'):
                update_freq = model.update_freq
            else:
                update_freq = model.module.update_freq
            total_loss /= update_freq

        if amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        losses_info = detach_loss_value(losses)
        losses_info.update({"iter_loss": total_loss.item()})
        pbar.set_postfix(losses_info)
        if (batch_id + 1) % update_freq == 0:
            if amp:
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()
