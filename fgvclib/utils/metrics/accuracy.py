import torch

def accuracy(outputs, targets, k=1):
    total = targets.size(0)
    _, predicts = outputs.topk(k, 1)
    corrects = predicts.eq(targets.data).cpu().sum()
    return torch.true_divide(100 * corrects, total)