import typing as t
import fiftyone as fo
from tqdm import tqdm

from fgvclib.datasets import Dataset_AnnoFolder
from fgvclib.configs import FGVCConfig

class VOXEL():

    def __init__(self, dataset, name:str, persistent=False) -> None:
        self.dataset = dataset
        self.name = name
        if self.name not in fo.list_datasets():
            self.fiftyone_dataset = self.create_dataset()
        else:
            self.fiftyone_dataset = fo.load_dataset(self.name)

    def create_dataset(self, ):
        return fo.Dataset(self.name)


    def load(self, ):
        
        samples = []

        for i in tqdm(range(len(self.dataset))):
            path, anno = self.dataset.get_imgpath_anno_pair(i)

            sample = fo.Sample(filepath=path)

            # Store classification in a field name of your choice
            sample["ground_truth"] = fo.Classification(label=anno)

            samples.append(sample)

            # Create dataset
        
        self.fiftyone_dataset.add_samples(samples)


    def launch(self, ):
        session = fo.launch_app()
        session.wait()

    def del_dataset(self, ):
        assert self.name in fo.list_datasets(), f"The dataset {self.name} does not exists"


    
