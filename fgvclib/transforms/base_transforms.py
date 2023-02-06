from torchvision import transforms
from PIL import Image

from fgvclib.transforms import transform
    
@transform("resize")
def resize(cfg: dict):
    return transforms.Resize(size=cfg['size'], interpolation=Image.BILINEAR)

@transform("random_crop")
def random_crop(cfg: dict):
    return transforms.RandomCrop(size=cfg['size'], padding=cfg['padding'])

@transform("center_crop")
def center_crop(cfg: dict):
    return transforms.CenterCrop(size=cfg['size'])

@transform("random_horizontal_flip")
def random_horizontal_flip(cfg: dict):
    return transforms.RandomHorizontalFlip(p=cfg['prob'])

@transform("to_tensor")
def to_tensor(cfg: dict):
    return transforms.ToTensor()

@transform("normalize")
def normalize(cfg: dict):
    return transforms.Normalize(mean=cfg['mean'], std=cfg['std'])

@transform("color_jitter")
def color_jitter(cfg: dict):
    return transforms.ColorJitter(brightness=cfg['brightness'], saturation=cfg['saturation'])

@transform('randomApply_gaussianBlur')
def randomApply_gaussianBlur(cfg: dict):
    return transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=cfg['prob'])

@transform('randomAdjust_sharpness')
def randomAdjust_sharpness(cfg:dict):
    return transforms.RandomAdjustSharpness(sharpness_factor=cfg['sharpness_factor'], p=cfg['prob'])
