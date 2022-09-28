import random
from torch.autograd import Variable

from fgvclib.criterions import compute_loss_value    , detach_loss_value

def progressive_updating_with_jigsaw(model, train_data, optimizer, use_cuda=True):
    inputs, targets = train_data
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
        inputs_ = jigsaw_generator(inputs, 2 ** (step_num - (step + 1)))
        _, losses = model(inputs_, targets, step)
        step_loss = compute_loss_value(losses)
        total_loss += step_loss.item()
        losses_info.update(detach_loss_value(losses))
        step_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    losses_info.update({"iter_loss": total_loss / step_num})
    return losses_info

    
def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws