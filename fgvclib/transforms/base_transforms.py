from torchvision import transforms
from PIL import Image

def resize(cfg):
    return transforms.Resize(size=cfg['size'], interpolation=Image.BILINEAR)

def random_crop(cfg):
    return transforms.RandomCrop(size=cfg['size'], padding=cfg['padding'])

def center_crop(cfg):
    return transforms.CenterCrop(size=cfg['size'])

def random_horizontal_flip(cfg):
    return transforms.RandomHorizontalFlip(p=cfg['prob'])

def to_tensor(cfg):
    return transforms.ToTensor()

def normalize(cfg):
    return transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
    