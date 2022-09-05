from .wandb_logger import WandbLogger
from .txt_logger import TxtLogger


__all__ = ["WandbLogger", "TxtLogger"]

def get_logger(logger_name):
    """Return the update strategy with the given name."""
    if logger_name not in globals():
        raise NotImplementedError(f"Logger not found: {logger_name}\nAvailable loggers: {__all__}")
    return globals()[logger_name]
