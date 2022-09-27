from torch import nn, Tensor
from torchmetrics import Metric, Accuracy, Precision, Recall


class NamedMetric(Metric):
    
    def __init__(self, name:str, metric:Metric):
        super(NamedMetric, self).__init__()
        self.name = name
        self.metric = metric
    
    def update(self, preds, targets):
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute()

    def forward(self, preds, targets) -> Tensor:
        return self.metric(preds, targets)

def accuracy(name:str="accuracy(top-1)", top_k:int=1, threshold:float=None) -> Metric:
    metric = Accuracy(top_k=top_k, threshold=threshold)
    return NamedMetric(name=name, metric=metric)

def precision(name:str="precision(threshold=0.5)", top_k:int=None, threshold:float=0.5) -> Metric:
    metric = Precision(top_k=top_k, threshold=threshold)
    return NamedMetric(name=name, metric=metric)

def recall(name:str="recall(threshold=0.5)", top_k:int=None, threshold:float=0.5) -> Metric:
    metric = Recall(top_k=top_k, threshold=threshold)
    return NamedMetric(name=name, metric=metric)
