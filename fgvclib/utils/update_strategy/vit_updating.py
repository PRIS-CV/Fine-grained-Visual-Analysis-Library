import numpy as np
import torch
from torch.autograd import Variable
import typing as t
from fgvclib.criterions import compute_loss_value, detach_loss_value


def vit_updating(model, train_data, optimizer, scheduler, use_cuda=True) -> t.Dict:
    inputs, targets = train_data
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    out, losses = model(inputs, targets)
    total_loss = compute_loss_value(losses)
    total_loss = total_loss.mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    losses_info = detach_loss_value(losses)
    losses_info.update({"iter_loss": total_loss.item()})

    return losses_info
