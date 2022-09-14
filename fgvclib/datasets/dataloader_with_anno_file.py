import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset



class Dataset_AnnoFile(Dataset):
    def __init__(self, dir_path, anno_path, train=False, transform=None, positive=None):
        self.dir_path = dir_path
        self.annotations = pd.read_csv(anno_path, sep=" ", header=None,
                              names=['ImageName', 'label'])
        self.image_names = self.annotations['ImageName'].tolist()
        self.labels = self.annotations['label'].tolist()

        self.train = train
        self.transform = transform

        self.positive = positive

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = os.path.join(self.dir_path, self.image_names[item])
        image = self.pil_loader(img_path)
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        if self.positive != None:
            positive_image = self.fetch_positive(self.positive, label, self.image_names[item])
            return image, positive_image, label
        return image, label
    
    def get_imgpath_anno_pair(self, idx):
        img_path = os.path.join(self.dir_path, self.image_names[idx])
        label = self.labels[idx]
        return img_path, label

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def fetch_positive(self, positive, label, path):
        other_img_info = self.annotations[(self.annotations.label == label) & (self.annotations.ImageName != path)]
        other_img_info = other_img_info.sample(min(positive, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.dir_path, e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        positive_img = [self.transform(img) for img in other_img]
        return positive_img
