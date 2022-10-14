from .dataloader_with_anno_folder import Dataset_AnnoFolder
from .dataloader_with_anno_file import Dataset_AnnoFile

from .datasets import FGVCDataset, available_datasets, CUB_200_2011

__all__ = [
    'Dataset_AnnoFolder', 'Dataset_AnnoFile', 'CUB_200_2011'
]


def get_dataset(dataset_name) -> FGVCDataset:
    r"""Return the dataset with the given name.

        Args: 
            dataset_name (str): 
                The name of dataset.
        
        Return: 
            The dataset contructor method.
    """

    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset {dataset_name} not found!\nAvailable datasets: {available_datasets()}")
    return globals()[dataset_name]
