# Copyright (c) 2022-present, BUPT-PRIS.

"""
    This file provides a api for evaluating FGVC algorithms.
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import typing as t
from fgvclib.metrics.metrics import NamedMetric


def evaluate_model(model:nn.Module, p_bar:t.Iterator, metrics:t.List[NamedMetric], use_cuda:bool=True) -> t.Dict:
    
    model.eval()
    results = dict()
    
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(p_bar):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            for metric in metrics:
                _ = metric.update(model(inputs), targets) 
    
    for metric in metrics:
        result = metric.compute()
        results.update({
            metric.name: round(result.item(), 3)
        })

    return results
