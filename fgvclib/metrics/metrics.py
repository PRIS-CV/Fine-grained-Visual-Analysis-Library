from torch import nn, Tensor
from torchmetrics import Metric, Accuracy, Precision, Recall


class NamedMetric(nn.Module):
    
    def __init__(self, name:str, metric:Metric):
        super(NamedMetric, self).__init__()
        self.name = name
        self.metric = metric

    def forward(self, preds, targets) -> Tensor:
        return self.metric(preds, targets)

def accuracy(top_k:int=1, threshold:float=None) -> Metric:
    return Accuracy(top_k=top_k, threshold=threshold)

def precision(top_k:int=None, threshold:float=0.5) -> Metric:
    return Precision(top_k=top_k, threshold=threshold)

def recall(top_k:int=None, threshold:float=0.5) -> Metric:
    return Recall(top_k=top_k, threshold=threshold)

