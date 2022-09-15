import argparse
import os
import torch
from tqdm import tqdm

from fgvclib.apis import *
from fgvclib.configs import FGVCConfig
from fgvclib.utils.lr_schedules import cosine_anneal_schedule
from fgvclib.datasets import Dataset_AnnoFolder
from fgvclib.utils.visualization import VOXEL


def main(cfg):
    
    model = build_model(cfg.MODEL)

    if cfg.RESUME_WEIGHT:
        assert os.path.exists(cfg.RESUME_WEIGHT), f"The resume weight {cfg.RESUME_WEIGHT} dosn't exists."
        model.load_state_dict(torch.load(cfg.RESUME_WEIGHT, map_location="cpu"))

    if cfg.USE_CUDA:
        assert torch.cuda.is_available(), f"Cuda is not available."
        model = torch.nn.DataParallel(model)
    
    train_transforms = build_transforms(cfg.TRANSFORMS.TRAIN)
    test_transforms = build_transforms(cfg.TRANSFORMS.TEST)

    train_loader = build_dataset(root=os.path.join(cfg.DATASETS.ROOT, 'train'), cfg=cfg.DATASETS.TRAIN, transforms=train_transforms)
    test_loader = build_dataset(root=os.path.join(cfg.DATASETS.ROOT, 'test'), cfg=cfg.DATASETS.TEST, transforms=test_transforms)

    optimizer = build_optimizer(cfg.OPTIMIZER, model)

    logger = build_logger(cfg)

    for epoch in range(cfg.START_EPOCH, cfg.EPOCH_NUM):
        
        train_bar = tqdm(train_loader)
        train_bar.set_description(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Training')

        logger(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Training')

        cosine_anneal_schedule(optimizer, epoch, cfg.EPOCH_NUM)
        update_model(model, optimizer, train_bar, strategy=cfg.UPDATE_STRATEGY, use_cuda=cfg.USE_CUDA, logger=logger)
        
        test_bar = tqdm(test_loader)
        test_bar.set_description(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Testing ')
        
        logger(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Testing ')

        acc = evaluate_model(model, test_bar, metrics=cfg.METRICS, use_cuda=cfg.USE_CUDA)
        logger("Evalution Result:")
        logger(acc)
    
    save_model(cfg=cfg, model=model.module, logger=logger)
    logger.finish()

def predict(cfg):

    model = build_model(cfg.MODEL)

    if cfg.RESUME_WEIGHT:
        assert os.path.exists(cfg.RESUME_WEIGHT), f"The resume weight {cfg.RESUME_WEIGHT} dosn't exists."
        model.load_state_dict(torch.load(cfg.RESUME_WEIGHT, map_location="cpu"))

    if cfg.USE_CUDA:
        assert torch.cuda.is_available(), f"Cuda is not available."
        model = torch.nn.DataParallel(model)

    transforms = build_transforms(cfg.TRANSFORMS.TEST)
    loader = build_dataset(root=os.path.join(cfg.DATASETS.ROOT, 'test'), cfg=cfg.DATASETS.TEST, transforms=transforms)
    dataset = loader.dataset

    voxel = VOXEL(dataset=dataset, name=cfg.FIFTYONE.NAME)
    voxel.load()
    voxel.predict(model, transforms, 100, cfg.MODEL.NAME)
    voxel.launch()
    
    

    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, help='the path of configuration file')
    args = parser.parse_args()

    config = FGVCConfig()
    config.load(args.config)
    cfg = config.cfg

    print(cfg)
    # main(cfg)
    predict(cfg)

    
