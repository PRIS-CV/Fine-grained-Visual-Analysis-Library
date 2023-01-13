import os.path as op
from PIL import Image
import gzip, tarfile
import typing as t

from torch.utils.data.dataset import Dataset


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


