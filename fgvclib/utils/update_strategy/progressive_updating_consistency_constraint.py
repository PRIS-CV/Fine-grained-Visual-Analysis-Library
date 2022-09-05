import random
import torch
import torch.nn as nn
from torch.autograd import Variable

BLOCKS = [[8, 8, 0, 0], [4, 4, 4, 0], [2, 2, 2, 2]]
alpha = [0.01, 0.05, 0.1]

def progressive_updating_consistency_constraint(model, train_data, optimizer, use_cuda=True):
    inputs, positive_inputs, targets = train_data
    batch_size = inputs.size(0)
    positive_inputs = positive_inputs[0]
    inputs = torch.cat([inputs, positive_inputs], 0)
    targets = torch.cat([targets, targets], 0)
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
    try:
        step_num = model.outputs_num
    except Exception:
        step_num = model.module.outputs_num
    for step in range(step_num):
        outputs, features = model(inputs, BLOCKS[step])
        outputs, features = outputs[step], features[step]
        loss = criterion1(outputs, targets) + criterion2(features[:batch_size], features[batch_size:]) * alpha[step]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()