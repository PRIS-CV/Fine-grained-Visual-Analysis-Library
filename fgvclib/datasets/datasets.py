import os
import os.path as op
from PIL import Image
import gzip, tarfile
import typing as t
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

    def _load_samples(self, ):
        pass

    def _load_categories(self) -> t.Union[dict, dict]:
        category2index = dict()
        index2category = dict()
        return category2index, index2category

    def _pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    
    def _download(self):
        pass
    
    def encode_category(self, category:str):
        return self.category2index[category]

    def decode_category(self, index:int):
        return self.index2category[index]

    def get_categories(self, ):
        return list(self.category2index.keys())

    def _extract_file(self, package:str):
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
    annotation_file: str = "CUB_200_2011/CUB_200_2011/classes.txt"
    image_dir: str = "CUB_200_2011/CUB_200_2011/images/"
    split_file: str = "CUB_200_2011/CUB_200_2011/train_test_split.txt"
    images_list_file: str = "CUB_200_2011/CUB_200_2011/images.txt" 

    def __init__(self, root:str, mode:str, download:bool=False, transforms:T.Compose=None, positive:int=None):
        assert mode in ["train", "test"], "The train"
        self.root = root
        if download:
            self._download()
        assert op.exists(op.join(self.root, "CUB_200_2011")), "Please download the dataset by setting download=True."
        self.category2index, self.index2category = self._load_categories()
        self.samples = self._load_samples(split=mode)

    def __getitem__(self, index:int):

        image = self.samples[index]
        label = image.split()[1]
        if self.transforms:
            image = self.transforms(image)
        if self.positive != None:
            positive_image = self._fetch_positive(self.positive, label, self.image_names[index])
            return image, positive_image, label
        
        return image, label

    def _load_samples(self, split) -> t.List[str]:
        image_ids = []
        samples = []
        mode = '1' if split == "train" else '0'
        with open(op.join(self.root, self.split_file)) as f:
            lines = f.readlines()
        for line in lines:
            image_id, is_train = line.split()
            if mode == is_train:
                image_ids.append(image_id)

        with open(op.join(self.root, self.split_file)) as f:
            lines = f.readlines()
        
        for line in lines:
            image_id, image_path = line.split()
            if image_id in image_ids:
                samples.append(op.join(self.root, self.image_dir, image_path))
        return samples

    def _load_categories(self) -> t.Union[dict, dict]:
        category2index = dict()
        index2category = dict()
        image_dir = op.join(self.root, self.image_dir) 
        sub_dirs = os.listdir(image_dir)
        for name in sub_dirs:
            index, category = name.split('.')
            category2index.update({category: int(index) - 1})
            index2category.update({int(index) - 1: category})
        return category2index, index2category
    
    def _download(self, overwrite=False):
        if op.exists(op.join(self.root, "CUB_200_2011")) and not overwrite:
            print("The dataset already exist!")
            return 
        if not op.exists(op.join(self.root, "CUB_200_2011.tgz")):
            wget.download(self.download_link, op.join(self.root, "CUB_200_2011.tgz"))
        self._extract_file(op.join(self.root, "CUB_200_2011.tgz"))
        print(f"{self.name} dataset is ready.")

    def _fetch_positive(self, positive, label, path):
        other_img_info = self.annotations[(self.annotations.label == label) & (self.annotations.ImageName != path)]
        other_img_info = other_img_info.sample(min(positive, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.dir_path, e['ImageName']) for e in other_img_info]
        other_img = [self._pil_loader(img) for img in other_img_path]
        positive_img = [self.transforms(img) for img in other_img]
        return positive_img
        

if __name__ == "__main__":
    print(available_datasets())
    dataset = CUB_200_2011("/data/wangxinran/dataset/", mode="test", download=True)
    print(len(dataset))
