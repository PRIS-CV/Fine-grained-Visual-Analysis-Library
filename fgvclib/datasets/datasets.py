import os
import os.path as op
from PIL import Image
import gzip, tarfile
import typing as t
import random
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import wget



def available_datasets():
    r"""Show all available FGVC datasets.

        Return:
            list: All available FGVC datasets.
    """

    return [subcls.__name__ for subcls in FGVCDataset.__subclasses__()]


class FGVCDataset(Dataset):
    name: str = "FGVCDataset"
    link: str = ""
    download_link:str = ""

    def __init__(self, root:str):
        self.root = root
        self.samples = self._load_samples()

    def __getitem__(self, index:int):
        return 
    
    def __len__(self):
        return len(self.samples)

    def _reverse_key_value(self, d:dict) -> dict:
        return {v:k for k, v in d.items()}

    def _load_annotations(self):
        pass

    def _load_samples(self, ):
        pass

    def _load_categories(self) -> t.Union[dict, list]:
        category2index = dict()
        index2category = list()
        return category2index, index2category

    def _pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    
    def _download(self):
        pass

    def _get_sample_by_category(self, catrgory, path):
        pass
    
    def encode_category(self, category:str):
        return self.category2index[category]

    def decode_category(self, index:int):
        return self.index2category[index]

    def get_categories(self, ):
        return list(self.category2index.keys())

    def _extract_file(self, package:str, delete:bool=True):
        if package.endswith('.gz'):
            self._un_gz(package)
        elif package.endswith('.tar'):
            pass
        elif package.endswith('.zip'):
            pass
        elif package.endswith('.tgz'):
            self._un_tgz(package)

    def _un_gz(self, package:str):
        
        gz_file = gzip.GzipFile(package)

        with open(op.basename(package), "w+") as f:
            f.write(gz_file.read()) 

        gz_file.close()

    def _un_tgz(self, package:str):
        tar = tarfile.open(package)
        tar.extractall(op.join(op.dirname(package), op.basename(package).split('.')[0]))


class CUB_200_2011(FGVCDataset):
    r"""The Caltech-UCSD Birds-200-2011 dataset.
    """

    name: str = "Caltech-UCSD Birds-200-2011"
    link: str = "http://www.vision.caltech.edu/datasets/cub_200_2011/"
    download_link: str = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    category_file: str = "CUB_200_2011/CUB_200_2011/classes.txt"
    annotation_file: str = "CUB_200_2011/CUB_200_2011/image_class_labels.txt"
    image_dir: str = "CUB_200_2011/CUB_200_2011/images/"
    split_file: str = "CUB_200_2011/CUB_200_2011/train_test_split.txt"
    images_list_file: str = "CUB_200_2011/CUB_200_2011/images.txt" 

    def __init__(self, root:str, mode:str, download:bool=False, transforms:T.Compose=None, positive:int=0):
        r"""
            The Caltech-UCSD Birds-200-2011 dataset class.
            Link: http://www.vision.caltech.edu/datasets/cub_200_2011/

            Args:
                root (str): 
                    The root directory of CUB dataset.
                
                mode (str):
                    The split of CUB dataset.
                
                download (bool):
                    Directly downloading CUB dataset by setting download=True. Default 
                    is False.
                
                transforms (torchvision.transforms.Compose):
                    The PyTorch transforms Compose class used to preprocessing the data.
                
                positive (int):
                    If positive = n > 0, the __getitem__ method will an extra list of n 
                    images of same category.
        """

        assert mode in ["train", "test"], "The mode of CUB datasets should be train or test"
        self.root = root
        if download:
            self._download()
        assert op.exists(op.join(self.root, "CUB_200_2011")), "Please download the dataset by setting download=True."
        assert positive >= 0, "The value of positive must larger or equal than 0"
        self.category2index, self.index2category = self._load_categories()
        self.positive = positive
        self.transforms = transforms
        self.annotations = self._load_annotations()
        self.samples, self.category2sample = self._load_samples(split=mode)

    def __getitem__(self, index:int):

        image_path, label = self.samples[index]
        image = self._pil_loader(image_path)
        
        if self.transforms:
            image = self.transforms(image)
        
        if self.positive > 0:
            positive_image = self._fetch_positive(self.positive, label, image_path)
            return image, positive_image, label
        
        return image, label

    def _load_annotations(self) -> dict:
        annotations = {}
        with open(op.join(self.root, self.annotation_file)) as f:
            lines = f.readlines()
        for line in lines:
            image_id, label = line.split()
            annotations[image_id] = int(label) - 1 
        return annotations
            
    def _load_samples(self, split) -> t.Union[t.List[str], dict]:
        image_ids = []
        samples = []
        mode = '1' if split == "train" else '0'
        with open(op.join(self.root, self.split_file)) as f:
            lines = f.readlines()
        for line in lines:
            image_id, is_train = line.split()
            if mode == is_train:
                image_ids.append(image_id)

        with open(op.join(self.root, self.images_list_file)) as f:
            lines = f.readlines()
        
        category2sample = {v: [] for v in self.category2index.values()}

        for line in lines:
            image_id, image_path = line.split()
            if image_id in image_ids:
                image_path = op.join(self.root, self.image_dir, image_path)
                label = self.annotations[image_id]
                sample = (image_path, label)
                samples.append(sample)
                category2sample[int(label)].append(image_path)
        
        return samples, category2sample

    def _load_categories(self) -> t.Union[dict, list]:
        category2index = dict()
        index2category = list()
        with open(op.join(self.root, self.category_file)) as f:
            lines = f.readlines()
        for line in lines:
            index, category = line.split()
            category2index[category] = int(index) - 1
            index2category.append(category)
        
        return category2index, index2category
    
    def _download(self, overwrite=False):
        if op.exists(op.join(self.root, "CUB_200_2011")) and not overwrite:
            print("Loading From Pre-download Dataset ...")
            return 
        if not op.exists(op.join(self.root, "CUB_200_2011.tgz")):
            print(f"Downloading {self.name} into {self.root} ...")
            wget.download(self.download_link, op.join(self.root, "CUB_200_2011.tgz"))
        self._extract_file(op.join(self.root, "CUB_200_2011.tgz"))
        os.remove(op.join(self.root, "CUB_200_2011.tgz"))
        print(f"{self.name} dataset is ready.")

    def _fetch_positive(self, positive, label, path):
        other_imgs = [sample_path for sample_path in self.category2sample[label] if sample_path != path]
        other_imgs = random.sample(other_imgs, min(positive, len(other_imgs)))
        other_imgs_path = [os.path.join(self.image_dir, oi) for oi in other_imgs]
        positive_imgs = [self._pil_loader(img) for img in other_imgs_path]
        if self.transforms:
            positive_imgs = [self.transforms(img) for img in positive_imgs]
        return positive_imgs
        