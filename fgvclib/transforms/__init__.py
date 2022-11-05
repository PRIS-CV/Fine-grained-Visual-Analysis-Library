from .base_transforms import *
from .mixup import mix_up
from .cutmix import cut_mix

__all__ = ["resize", "center_crop", "random_crop", "random_horizontal_flip", "to_tensor", "normalize", "mix_up", "cut_mix"]


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
