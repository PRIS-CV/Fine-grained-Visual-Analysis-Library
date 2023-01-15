from torch.autograd import Variable
import typing as t

from fgvclib.criterions import compute_loss_value, detach_loss_value


def general_updating(model, train_data, optimizer, use_cuda=True, **kwargs) -> t.Dict:
    inputs, targets = train_data
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    out, losses = model(inputs, targets)
    total_loss = compute_loss_value(losses)
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses_info = detach_loss_value(losses)
    losses_info.update({"iter_loss": total_loss.item()})
    
    return losses_info