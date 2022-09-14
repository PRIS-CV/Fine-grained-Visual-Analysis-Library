from .global_avg_pooling import GlobalAvgPooling
from .global_max_pooling import GlobalMaxPooling, MaxPool2d

__all__ = [
    'GlobalAvgPooling', 'GlobalMaxPooling', 'MaxPool2d'
]

def get_encoding(encoding_name):
    """Return the backbone with the given name."""
    if encoding_name not in globals():
        raise NotImplementedError(f"Encoding not found: {encoding_name}\nAvailable encodings: {__all__}")
    return globals()[encoding_name]