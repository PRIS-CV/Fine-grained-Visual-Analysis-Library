from torchmetrics import Metric
from .metrics import accuracy, precision, recall


__all__ = ["accuracy", "precision", "recall"]

def get_metric(metric_name) -> Metric:
    """Return the backbone with the given name."""
    if metric_name not in globals():
        raise NotImplementedError(f"Metric {metric_name} not found!\nAvailable metrics: {__all__}")
    return globals()[metric_name]
