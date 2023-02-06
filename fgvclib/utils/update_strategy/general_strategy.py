from torch.autograd import Variable
import typing as t
from torch.cuda.amp import autocast, GradScaler

from . import update_strategy
from fgvclib.criterions import compute_loss_value, detach_loss_value



@update_strategy("general_strategy")
def general_strategy(model, train_data, optimizer, use_cuda=True, amp=False, **kwargs) -> t.Dict:
    if amp:
        scaler = GradScaler()
    inputs, targets = train_data
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    if amp:
        with autocast():
            out, losses = model(inputs, targets)
            total_loss = compute_loss_value(losses)
            scaler.scale(total_loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
    else:
        out, losses = model(inputs, targets)
        total_loss = compute_loss_value(losses)
        total_loss.backward()
        optimizer.step()
    
    optimizer.zero_grad()
    losses_info = detach_loss_value(losses)
    losses_info.update({"iter_loss": total_loss.item()})
    
    return losses_info