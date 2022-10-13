import os
import os.path as op
import typing as t
from torch.utils.data.dataset import Dataset
import wget
import gzip, tarfile


def available_datasets():
    return 


class FGVCDataset(Dataset):
    name: str = "FGVCDataset"
    link: str = ""
    download_link:str = ""

    def __init__(self, root:str):
        self.root = root

    def __getitem__(self, index:int):
        return 

    def _reverse_key_value(self, d:dict) -> dict:
        return {v:k for k, v in d.items()}

    def load_samples(self, ):
        pass

    def _load_categories(self) -> t.Union[dict, dict]:
        category2index = dict()
        index2category = dict()
        return category2index, index2category
    
    def download(self):
        print(f"Download {self.name} dataset into {self.root}")
    
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
    name: str = "Caltech-UCSD Birds-200-2011"
    link: str = "http://www.vision.caltech.edu/datasets/cub_200_2011/"
    download_link: str = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    annotation_file: str = "CUB_200_2011/CUB_200_2011/classes.txt"
    image_dir: str = "CUB_200_2011/CUB_200_2011/images/"
    split_file: str = "CUB_200_2011/CUB_200_2011/train_test_split.txt"

    def __init__(self, root:str, mode:str):
        assert mode in ["train", "test"], "The train"
        self.root = root
        self.category2index, self.index2category = self._load_categories()
        self.train_samples, test_samples = self._load_samples(split=mode)

    def __getitem__(self, index:int):
        return 

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
    
    def download(self, overwrite=False):
        if op.exists(op.join(self.root, "CUB_200_2011")) and not overwrite:
            print("The dataset already exist!")
            return 
        if not op.exists(op.join(self.root, "CUB_200_2011.tgz")):
            wget.download(self.download_link, op.join(self.root, "CUB_200_2011.tgz"))
        self._extract_file(op.join(self.root, "CUB_200_2011.tgz"))
        print(f"{self.name} dataset is ready.")
        

if __name__ == "__main__":
    dataset = CUB_200_2011("/data/wangxinran/dataset/")
    dataset.download()
    print(dataset.get_categories())
    print(dataset.encode_category('Least_Flycatcher'))
