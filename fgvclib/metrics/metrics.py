from torch import nn, Tensor
from torchmetrics import Metric, Accuracy, Precision, Recall

from fgvclib.metrics import metric 


class NamedMetric(Metric):
    r"""Metric item with name, used for evaluation.  
    """
    
    def __init__(self, name:str, metric:Metric):
        r"""The initalization of A NamedMetirc object.
            Args:
                name (str): 
                    The name of metric, e.g. accuracy(top-1)
                metric (Metric): 
                    The Metric object of torchmetrics
        """
        super(NamedMetric, self).__init__()
        self.name = name
        self.metric = metric
    
    def update(self, preds, targets):
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute()

    def forward(self, preds, targets) -> Tensor:
        return self.metric(preds, targets)


@metric("accuracy")
def accuracy(name:str="accuracy(top-1)", top_k:int=1, threshold:float=None) -> Metric:
    r"""The accuracy metric constructor, for details about the meanings of the parameters, see torchmetrics.Accuracy object.
            Args:
                name (str): 
                    The name of metric, e.g. accuracy(top-1)
                
                top_k (int): 
                    Number of the highest probability or logit score predictions considered finding the correct label.
                
                threshhold (float, optional): 
                    Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
                    of binary or multi-label inputs.

            Return:
                NamedMetirc: A torchmetrics metric with customed name.
    """

    metric = Accuracy(top_k=top_k, threshold=threshold)
    return NamedMetric(name=name, metric=metric)


@metric("precision")
def precision(name:str="precision(threshold=0.5)", top_k:int=None, threshold:float=0.5) -> Metric:
    r"""The precision metric constructor, for details about the meanings of the parameters, see torchmetrics.Precision object.
            Args:
                name (str): 
                    The name of metric, e.g. accuracy(top-1)
                
                top_k (int): 
                    Number of the highest probability or logit score predictions considered finding the correct label.
                
                threshhold (float, optional): 
                    Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
                    of binary or multi-label inputs.

            Return:
                NamedMetirc: A torchmetrics metric with customed name.
    """

    metric = Precision(top_k=top_k, threshold=threshold)
    return NamedMetric(name=name, metric=metric)


@metric("recall")
def recall(name:str="recall(threshold=0.5)", top_k:int=None, threshold:float=0.5) -> Metric:
    r"""The recall metric constructor, for details about the meanings of the parameters, see torchmetrics.Recall object.
            Args:
                name (str): 
                    The name of metric, e.g. accuracy(top-1)
                
                top_k (int): 
                    Number of the highest probability or logit score predictions considered finding the correct label.
                
                threshhold (float, optional): 
                    Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
                    of binary or multi-label inputs.

            Return:
                NamedMetirc: A torchmetrics metric with customed name.
    """

    metric = Recall(top_k=top_k, threshold=threshold)
    return NamedMetric(name=name, metric=metric)
