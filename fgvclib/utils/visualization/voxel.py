import typing as t
import fiftyone as fo
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as func
import torch.nn as nn
import torch

from fgvclib.datasets import Dataset_AnnoFolder
from fgvclib.configs import FGVCConfig

class VOXEL():

    def __init__(self, dataset, name:str, persistent:bool=False, cuda:bool=True) -> None:
        self.dataset = dataset
        self.name = name
        if self.name not in self.loaded_datasets():
            self.fo_dataset = self.create_dataset()
        else:
            self.fo_dataset = fo.load_dataset(self.name)
        self.persistent = persistent
        self.cuda = cuda

    def create_dataset(self, ):
        return fo.Dataset(self.name)

    def loaded_datasets(self):
        return fo.list_datasets()

    def load(self, ):
        
        samples = []

        for i in tqdm(range(len(self.dataset))):
            path, anno = self.dataset.get_imgpath_anno_pair(i)

            sample = fo.Sample(filepath=path)

            # Store classification in a field name of your choice
            sample["ground_truth"] = fo.Classification(label=anno)

            samples.append(sample)

            # Create dataset
        
        self.fo_dataset.add_samples(samples)
        self.fo_dataset.persistent = self.persistent

    def predict(self, model:nn.Module, transforms, n, name="prediction", seed=51):
        model.eval()
        predictions_view = self.fo_dataset.take(n, seed=seed)

        with fo.ProgressBar() as pb:
            for sample in pb(predictions_view):
                image = Image.open(sample.filepath)
                if self.cuda:
                    image = transforms(image).cuda().unsqueeze(0)
                    pred = model(image)
                    index = torch.argmax(pred).item()
                    confidence = pred[:, index].item()

    
                sample[name] = fo.Classification(
                    label=str(index),
                    confidence=confidence
                )

                sample.save()
        print("Finished adding predictions")


    def launch(self, ):
        session = fo.launch_app()
        session.wait()

    def del_dataset(self, ):
        assert self.name in fo.list_datasets(), f"The dataset {self.name} does not exists"


    
