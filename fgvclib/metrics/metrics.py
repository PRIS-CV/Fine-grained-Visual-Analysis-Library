import torchmetrics
from torchmetrics import Metric, Accuracy, Precision, Recall
import torch


def accuracy(top_k:int=None, num_classes:int=None) -> Metric:
    return Accuracy(top_k=top_k, num_classes=num_classes)

def precision(top_k:int=None, num_classes:int=None) -> Metric:
    return Precision(top_k=top_k, num_classes=num_classes)

def recall(top_k:int=None, num_classes:int=None):
    return Recall(top_k=top_k, num_classes=num_classes)







