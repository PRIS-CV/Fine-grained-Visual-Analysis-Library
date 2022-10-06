import typing as t
import torch
from torch import nn, Tensor
from torch.autograd import Variable

from fgvclib.criterions import compute_loss_value, detach_loss_value

BLOCKS = [[8, 8, 0, 0], [4, 4, 4, 0], [2, 2, 2, 2]]
alpha = [0.01, 0.05, 0.1]

def progressive_updating_consistency_constraint(model:nn.Module, train_data:t.Tuple[Tensor, Tensor, Tensor], optimizer, use_cuda=True) -> t.Dict:
    inputs, positive_inputs, targets = train_data
    batch_size = inputs.size(0)
    positive_inputs = positive_inputs[0]
    inputs = torch.cat([inputs, positive_inputs], 0)
    targets = torch.cat([targets, targets], 0)
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    losses_info = {}
    total_loss = 0.
    
    try:
        step_num = model.outputs_num
    except Exception:
        step_num = model.module.outputs_num
    for step in range(step_num):
        _, losses = model(inputs, targets, step)
        
        step_loss = compute_loss_value(losses)
        total_loss += step_loss.item()
        losses_info.update(detach_loss_value(losses))
        step_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    losses_info.update({"iter_loss": total_loss / step_num})
    return losses_info
