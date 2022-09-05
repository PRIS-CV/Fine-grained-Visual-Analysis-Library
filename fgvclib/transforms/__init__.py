from .base_transforms import *

__all__ = ["resize", "center_crop", "random_crop", "random_horizontal_flip", "to_tensor", "normalize"]


def get_transform(transform_name):
    """Return the backbone with the given name."""
    if transform_name not in globals():
        raise NotImplementedError(f"Transform not found: {transform_name}\nAvailable transforms: {__all__}")
    return globals()[transform_name]

