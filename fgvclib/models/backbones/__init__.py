from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from .vgg import vgg11, vgg13, vgg16, vgg19

__all__ = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_bc', 'resnet101_bc',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
]


def get_backbone(backbone_name):
    """Return the backbone with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError(f"Backbone {backbone_name} not found!\nAvailable backbones: {__all__}")
    return globals()[backbone_name]

