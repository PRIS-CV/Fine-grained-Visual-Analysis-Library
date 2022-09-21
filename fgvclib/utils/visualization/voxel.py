from math import inf
import typing as t
import fiftyone as fo
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch

from fgvclib.utils.interpreter import Interpreter

class VOXEL:

    def __init__(self, dataset, name:str, persistent:bool=False, cuda:bool=True, interpreter:Interpreter=None) -> None:
        self.dataset = dataset
        self.name = name
        self.persistent = persistent
        self.cuda = cuda
        self.interpreter = interpreter

        if self.name not in self.loaded_datasets():
            self.fo_dataset = self.create_dataset()
            self.load()
        else:
            self.fo_dataset = fo.load_dataset(self.name)

        self.view = self.fo_dataset.view() 

    def create_dataset(self) -> fo.Dataset:
        return fo.Dataset(self.name)

    def loaded_datasets(self) -> t.List:
        return fo.list_datasets()

    def load(self):
        
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

    def predict(self, model:nn.Module, transforms, n:int=inf, name="prediction", seed=51):
        model.eval()
        if n < inf:
            self.view = self.fo_dataset.take(n, seed=seed)

        with fo.ProgressBar() as pb:
            for sample in pb(self.view):
                image = Image.open(sample.filepath)
                image = transforms(image).unsqueeze(0)
                
                if self.cuda:
                    image = image.cuda()
                    pred = model(image)
                    index = torch.argmax(pred).item()
                    confidence = pred[:, index].item()

    
                sample[name] = fo.Classification(
                    label=str(index),
                    confidence=confidence
                )

                if self.interpreter:
                    heatmap = self.interpreter(image_path=sample.filepath, image_tensor=image, transforms=transforms)
                    sample["heatmap"] = fo.Heatmap(map=heatmap)

                sample.save()
        print("Finished adding predictions")

    
    def delete_field(self, field_name: str):
        self.fo_dataset.delete_frame_field(field_name=field_name)


    def launch(self) -> None:
        session = fo.launch_app(self.view)
        session.wait()

    def del_dataset(self):
        assert self.name in fo.list_datasets(), f"The dataset {self.name} does not exists"


    
