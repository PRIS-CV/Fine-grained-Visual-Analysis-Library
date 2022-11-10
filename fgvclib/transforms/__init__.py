from .base_transforms import *
from .mixup import MixUp
from .cutmix import CutMix

__all__ = ["resize", "center_crop", "random_crop", "random_horizontal_flip", "to_tensor", "normalize"]


def get_transform(transform_name):
    r"""Return the transform with the given name.

        Args: 
            transform_name (str): 
                The name of interpreter.
        
        Return: 
            The transform contructor method.
    """

    if transform_name not in globals():
        raise NotImplementedError(f"Transform not found: {transform_name}\nAvailable transforms: {__all__}")
    return globals()[transform_name]
