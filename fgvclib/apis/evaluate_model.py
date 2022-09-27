import torch
from torch.autograd import Variable
import typing as t
from fgvclib.metrics.metrics import NamedMetric

from ..utils.metrics import *

def evaluate_model(model, p_bar:t.Iterator, metrics:t.List[NamedMetric], use_cuda:bool=True) -> t.Dict:
    
    model.eval()
    targets_all = []
    outputs_all = []
    results = dict()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(p_bar):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            # outputs_all.append(model(inputs, None))
            # targets_all.append(targets)
            for metric in metrics:
                _ = metric.update(model(inputs), targets) 

    # targets_all, outputs_all = torch.cat(targets_all, 0), torch.cat(outputs_all, 0)
    
    for metric in metrics:
        result = metric.compute()
        results.update({
            metric.name: round(result.item(), 3)
        })
        
        # result = get_result(outputs_all, targets_all, metric)
        # results.update({
        #     metric: round(result.item(), 3)
        # })

    return results

def get_result(outputs, targets, metric):
    if metric == "top1-accuracy":
        return accuracy(outputs, targets, k=1)
    elif metric == "top5-accuracy":
        return accuracy(outputs, targets, k=5)