from torchvision import transforms
from PIL import Image

def resize(cfg: dict):
    return transforms.Resize(size=cfg['size'], interpolation=Image.BILINEAR)

def random_crop(cfg: dict):
    return transforms.RandomCrop(size=cfg['size'], padding=cfg['padding'])

def center_crop(cfg: dict):
    return transforms.CenterCrop(size=cfg['size'])

def random_horizontal_flip(cfg: dict):
    return transforms.RandomHorizontalFlip(p=cfg['prob'])

def to_tensor(cfg: dict):
    return transforms.ToTensor()

def normalize(cfg: dict):
    return transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
