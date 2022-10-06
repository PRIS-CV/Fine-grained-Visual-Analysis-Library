# Copyright (c) 2022-present, BUPT-PRIS.

"""
    build.py provides various apis for building a training or evaluation system fast.
"""

import torch
from torch import nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torch.utils.data import DataLoader
import typing as t
from yacs.config import CfgNode

from fgvclib.configs.utils import turn_list_to_dict as tltd
from fgvclib.criterions import get_criterion
from fgvclib.datasets import Dataset_AnnoFolder
from fgvclib.metrics import get_metric
from fgvclib.models.sotas import get_model
from fgvclib.models.backbones import get_backbone
from fgvclib.models.encoders import get_encoding
from fgvclib.models.necks import get_neck
from fgvclib.models.heads import get_head
from fgvclib.transforms import get_transform
from fgvclib.utils.logger import get_logger, Logger
from fgvclib.utils.interpreter import get_interpreter, Interpreter
from fgvclib.metrics import NamedMetric


def build_model(model_cfg: CfgNode) -> nn.Module:
    r"""Build a FGVC model according to config.

    Args:
        model_cfg (CfgNode): The model config node of root config.
    Returns:
        nn.Module: The FGVC model.
    """

    backbone_builder = get_backbone(model_cfg.BACKBONE.NAME)
    backbone = backbone_builder(cfg=tltd(model_cfg.BACKBONE.ARGS))

    if model_cfg.ENCODING.NAME:
        encoding_builder = get_encoding(model_cfg.ENCODING.NAME)
        encoding = encoding_builder(cfg=tltd(model_cfg.ENCODING.ARGS))
    else:
        encoding = None

    if model_cfg.NECKS.NAME:
        neck_builder = get_neck(model_cfg.NECKS.NAME)
        necks = neck_builder(cfg=tltd(model_cfg.NECKS.ARGS))
    else:
        necks = None

    head_builder = get_head(model_cfg.HEADS.NAME)
    heads = head_builder(class_num=model_cfg.CLASS_NUM, cfg=tltd(model_cfg.HEADS.ARGS))

    criterions = {}
    for item in model_cfg.CRITERIONS:
        criterions.update({item["name"]: {"fn": build_criterion(item), "w": item["w"]}})
    
    model_builder = get_model(model_cfg.NAME)
    model = model_builder(backbone=backbone, encoding=encoding, necks=necks, heads=heads, criterions=criterions)
    
    return model

def build_logger(cfg: CfgNode) -> Logger:
    r"""Build a Logger object according to config.

    Args:
        cfg (CfgNode): The root config node.
    Returns:
        Logger: The Logger object.
    """

    return get_logger(cfg.LOGGER.NAME)(cfg)

def build_transforms(transforms_cfg: CfgNode) -> transforms.Compose:
    r"""Build transforms for train or test dataset according to config.

    Args:
        transforms_cfg (CfgNode): The root config node.
    Returns:
        transforms.Compose: The transforms.Compose object in Pytorch.
    """

    return transforms.Compose([get_transform(item['name'])(item) for item in transforms_cfg])

def build_dataset(root:str, cfg: CfgNode, transforms) -> DataLoader:
    r"""Build a dataloader for training or evaluation.

    Args:
        root (str): The directory of dataset.
        cfg (CfgNode): The root config node.
    Returns:
        DataLoader: A Pytorch Dataloader.
    """

    dataset = Dataset_AnnoFolder(root=root, transform=transforms, positive=cfg.POSITIVE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE, num_workers=cfg.NUM_WORKERS)

    return data_loader

def build_optimizer(optim_cfg: CfgNode, model:t.Union[nn.Module, nn.DataParallel]) -> Optimizer:
    r"""Build a optimizer for training.

    Args:
        optim_cfg (CfgNode): The optimizer config node of root config node.
    Returns:
        Optimizer: A Pytorch Optimizer.
    """

    params= list()
    model_attrs = ["backbone", "encoding", "necks", "heads"]

    if isinstance(model, nn.DataParallel):
        for attr in model_attrs:
            if getattr(model.module, attr) and optim_cfg.LR[attr]:
                params.append({
                    'params': getattr(model.module, attr).parameters(), 
                    'lr': optim_cfg.LR[attr]
                })
                print(attr, optim_cfg.LR[attr])
    else:
        for attr in model_attrs:
            if getattr(model, attr) and optim_cfg.LR[attr]:
                params.append({
                    'params': getattr(model, attr).parameters(), 
                    'lr': optim_cfg.LR[attr]
                })
    optimizer = optim.SGD(params=params, momentum=optim_cfg.MOMENTUM, weight_decay=optim_cfg.WEIGHT_DECAY)
    
    return optimizer

def build_criterion(criterion_cfg: CfgNode) -> nn.Module:
    r"""Build loss function for training.

    Args:
        criterion_cfg (CfgNode): The criterion config node of root config node.
    Returns:
        nn.Module: A loss function.
    """

    criterion_builder = get_criterion(criterion_cfg['name'])
    criterion = criterion_builder(cfg=tltd(criterion_cfg['args']))
    return criterion

def build_interpreter(model: nn.Module, cfg: CfgNode) -> Interpreter:
    r"""Build loss function for training.

    Args:
        cfg (CfgNode): The root config node.
    Returns:
        Interpreter: A Interpreter.
    """

    return get_interpreter(cfg.INTERPRETER.NAME)(model, cfg)

def build_metrics(metrics_cfg: CfgNode, use_cuda:bool=True) -> t.List[NamedMetric]:
    r"""Build metrics for evaluation.

    Args:
        metrics_cfg (CfgNode): The metric config node of root config node.
    Returns:
        t.List[NamedMetric]: A List of NamedMetric.
    """

    metrics = []
    for cfg in metrics_cfg:
        metric = get_metric(cfg["metric"])(name=cfg["name"], top_k=cfg["top_k"], threshold=cfg["threshold"])
        if use_cuda:
            metric = metric.cuda()
        metrics.append(metric)
    return metrics


