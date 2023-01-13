# Copyright (c) 2022-present, BUPT-PRIS.

"""
    This file provides a api for evaluating FGVC algorithms.
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import typing as t
from fgvclib.metrics.metrics import NamedMetric


def evaluate_model(model:nn.Module, p_bar:t.Iterable, metrics:t.List[NamedMetric], use_cuda:bool=True) -> t.Dict:
    r"""Evaluate the FGVC model.

    Args:
        model (nn.Module): 
            The FGVC model.
        p_bar (iterable): 
            A iterator provide test data.
        metrics (List[NamedMetric]): 
            List of metrics. 
        use_cuda (boolean, optional): 
            Whether to use gpu. 
            
    Returns:
        dict: The result dict.
    """
    
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
