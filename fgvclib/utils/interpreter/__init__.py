from .interpreter import Interpreter
from .cam import cam


__all__ = ["cam"]

def get_interpreter(interpreter_name):
    """Return the update strategy with the given name."""
    if interpreter_name not in globals():
        raise NotImplementedError(f"Interpreter not found: {interpreter_name}\nAvailable loggers: {__all__}")
    return globals()[interpreter_name]

