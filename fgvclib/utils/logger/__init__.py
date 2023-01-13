from .logger import Logger
from .wandb_logger import wandb_logger
from .txt_logger import txt_logger


__all__ = ["wandb_logger", "txt_logger"]

def get_logger(logger_name):
    r"""Return the logger with the given name.

        Args: 
            logger_name (str): 
                The name of logger.
        
        Return: 
            The logger contructor method.
    """

    if logger_name not in globals():
        raise NotImplementedError(f"Logger not found: {logger_name}\nAvailable loggers: {__all__}")
    return globals()[logger_name]
