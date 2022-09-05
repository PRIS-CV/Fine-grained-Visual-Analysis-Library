from .multi_scale_conv import multi_scale_conv

__all__ = [
    'multi_scale_conv'
]

def get_neck(neck_name):
    """Return the backbone with the given name."""
    if neck_name not in globals():
        raise NotImplementedError(f"Neck not found: {neck_name}\nAvailable necks: {__all__}")
    return globals()[neck_name]