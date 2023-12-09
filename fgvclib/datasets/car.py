import os
import os.path as op
import random
import typing as t
from typing import List, Any, Tuple, Union

import torchvision.transforms as T
import wget
import pathlib
from fgvclib.datasets.datasets import FGVCDataset
from fgvclib.datasets import dataset
import scipy.io as sio


@dataset("StanfordCars")
class StanfordCars(FGVCDataset):
    r"""The Stanford Cars dataset.
    """

    name: str = "Stanford Cars"
    link: str = "https://ai.stanford.edu/~jkrause/cars/car_dataset.html"
    download_link: str = ""

    def __init__(self, root: str, mode: str, download: bool = False, transforms: T.Compose = None, positive:int=0):
        r"""
            Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>

            The Cars dataset contains 16,185 images of 196 classes of cars. The data is
            split into 8,144 training images and 8,041 testing images, where each class
            has been split roughly in a 50-50 split

            .. note::
              This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

            Args:
                root (str): 
                    The root directory of Stanford Cars dataset.
                
                mode (str):
                    The split of Stanford Cars dataset.
                
                download (bool):
                    Directly downloading Stanford Cars dataset by setting download=True. Default
                    is False.
                
                transforms (torchvision.transforms.Compose):
                    The PyTorch transforms Compose class used to preprocessing the data.
                

        """

        assert mode in ["train", "test"], "The mode of Stanford Cars datasets should be train or test"
        self.root = root
        if download:
            self._download()
        assert op.exists(op.join(self.root, "stanford_cars")), "Please download the dataset by setting download=True."
        assert positive >= 0, "The value of positive must larger or equal than 0"
        self.positive = positive
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if mode == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        self.transforms = transforms
        self.annotations = self._load_annotations()
        self.samples = self._load_samples()
        self.category2index, self.index2category = self._load_categories()

    def __getitem__(self, index: int):

        image_path, label = self.samples[index]
        image = self._pil_loader(image_path)

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def _load_annotations(self) -> List[Any]:
        annotations = []
        for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]:
            annotations.append(annotation)
        return annotations

    def _load_samples(self) -> List[Tuple[str, Union[int, Any]]]:
        samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in self.annotations
        ]
        return samples

    def _load_categories(self) -> Tuple[dict, Any]:
        index2category = sio.loadmat(str(self._base_folder / "devkit" / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        category2index = {cls: i for i, cls in enumerate(index2category)}
        return category2index, index2category

    def _download(self, overwrite=False):
        if op.exists(op.join(self.root, "stanford_cars")) and not overwrite:
            print("Loading From Pre-download Dataset ...")
            return
        if not op.exists(op.join(self.root, "stanford_cars.tgz")):
            print(f"Downloading {self.name} into {self.root} ...")
            wget.download(self.download_link, op.join(self.root, "stanford_cars.tgz"))
        self._extract_file(op.join(self.root, "stanford_cars.tgz"))
        os.remove(op.join(self.root, "stanford_cars.tgz"))
        print(f"{self.name} dataset is ready.")

    def _fetch_positive(self, positive, label, path):
        other_imgs = [sample_path for sample_path in self.category2sample[label] if sample_path != path]
        other_imgs = random.sample(other_imgs, min(positive, len(other_imgs)))
        other_imgs_path = [os.path.join(self.image_dir, oi) for oi in other_imgs]
        positive_imgs = [self._pil_loader(img) for img in other_imgs_path]
        if self.transforms:
            positive_imgs = [self.transforms(img) for img in positive_imgs]
        return positive_imgs

