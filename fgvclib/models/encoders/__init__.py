from .pooling import global_max_pooling, global_avg_pooling, max_pooling_2d

__all__ = [
    'global_avg_pooling', 'global_max_pooling', 'max_pooling_2d'
]

def get_encoding(encoding_name):
    """Return the backbone with the given name."""
    if encoding_name not in globals():
        raise NotImplementedError(f"Encoding not found: {encoding_name}\nAvailable encodings: {__all__}")
    return globals()[encoding_name]