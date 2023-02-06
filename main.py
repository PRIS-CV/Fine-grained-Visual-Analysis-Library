import argparse
import os
import torch
from tqdm import tqdm
from yacs.config import CfgNode


from fgvclib.apis import *
from fgvclib.configs import FGVCConfig
from fgvclib.utils import init_distributed_mode


def train(cfg: CfgNode):
    r"""Train and validate a FGVC algorithm.

    Args:
        cfg (CfgNode): The root config loaded by FGVCConfig object.
    """

    model = build_model(cfg.MODEL)
    print(model.get_structure())

    if cfg.RESUME_WEIGHT:
        assert os.path.exists(cfg.RESUME_WEIGHT), f"The resume weight {cfg.RESUME_WEIGHT} dosn't exists."
        model.load_state_dict(torch.load(cfg.RESUME_WEIGHT, map_location="cpu"))
    
    if cfg.USE_CUDA:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    train_transforms = build_transforms(cfg.TRANSFORMS.TRAIN)
    test_transforms = build_transforms(cfg.TRANSFORMS.TEST)

    train_set = build_dataset(
        name=cfg.DATASET.NAME, 
        root=cfg.DATASET.ROOT, 
        mode="train", 
        mode_cfg=cfg.DATASET.TRAIN, 
        transforms=train_transforms,
    )

    test_set = build_dataset(
        name=cfg.DATASET.NAME, 
        root=cfg.DATASET.ROOT, 
        mode="test", 
        mode_cfg=cfg.DATASET.TEST, 
        transforms=test_transforms
    )

    model.to(device)
    if cfg.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[cfg.GPU])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        sampler_cfg = cfg.SAMPLER
        train_sampler = build_sampler(sampler_cfg.TRAIN)(train_set, **tltd(sampler_cfg.TRAIN.ARGS))
        test_sampler = build_sampler(sampler_cfg.TEST)(test_set, **tltd(sampler_cfg.TEST.ARGS))
    
    train_loader = build_dataloader(
        dataset=train_set, 
        mode_cfg=cfg.DATASET.TRAIN,
        sampler=train_sampler,
        # is_batch_sampler=sampler_cfg.TRAIN.IS_BATCH_SAMPLER
        is_batch_sampler=False
    )

    test_loader = build_dataloader(
        dataset=test_set, 
        mode_cfg=cfg.DATASET.TEST,
        sampler=test_sampler,
        # is_batch_sampler=sampler_cfg.TEST.IS_BATCH_SAMPLER
        is_batch_sampler=False
    )

    optimizer = build_optimizer(cfg.OPTIMIZER, model)

    logger = build_logger(cfg)

    metrics = build_metrics(cfg.METRICS)

    lr_schedule = build_lr_schedule(optimizer, cfg.LR_SCHEDULE, train_loader)

    update_fn = build_update_function(cfg)

    evaluate_fn = build_evaluate_function(cfg)

    for epoch in range(cfg.START_EPOCH, cfg.EPOCH_NUM):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_bar = tqdm(train_loader)
        train_bar.set_description(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Training')

        logger(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Training')

        update_fn(
            model, optimizer, train_bar, 
            strategy=cfg.UPDATE_STRATEGY, use_cuda=cfg.USE_CUDA, lr_schedule=lr_schedule, 
            logger=logger, epoch=epoch, total_epoch=cfg.EPOCH_NUM, amp=cfg.AMP
        )
        
        test_bar = tqdm(test_loader)
        test_bar.set_description(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Testing ')
        
        logger(f'Epoch: {epoch + 1} / {cfg.EPOCH_NUM} Testing ')

        acc = evaluate_fn(model, test_bar, metrics=metrics, use_cuda=cfg.USE_CUDA)
        print(acc)
        logger("Evalution Result:")
        logger(acc)
    
    if cfg.DISTRIBUTED:
        model_with_ddp = model.module
    else:
        model_with_ddp = model
    save_model(cfg=cfg, model=model_with_ddp, logger=logger)
    logger.finish()

def predict(cfg: CfgNode):
    r"""Evaluate a FGVC algorithm.

    Args:
        cfg (CfgNode): The root config loaded by FGVCConfig object 
    """

    model = build_model(cfg.MODEL)
    weight_path = os.path.join(cfg.WEIGHT.SAVE_DIR, cfg.WEIGHT.NAME)
    assert os.path.exists(weight_path), f"The weight {weight_path} dosn't exists."
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict=state_dict)

    if cfg.USE_CUDA:
        assert torch.cuda.is_available(), f"Cuda is not available."
        model = torch.nn.DataParallel(model)

    transforms = build_transforms(cfg.TRANSFORMS.TEST)
    loader = build_dataset(root=os.path.join(cfg.DATASETS.ROOT, 'test'), cfg=cfg.DATASETS.TEST, transforms=transforms)
    
    pbar = tqdm(loader)
    metrics = build_metrics(cfg.METRICS)
    evaluate_fn = build_evaluate_function(cfg)
    acc = evaluate_fn(model, pbar, metrics=metrics, use_cuda=cfg.USE_CUDA)

    print(acc)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, help='the path of configuration file')
    parser.add_argument('--task', type=str, help='the path of configuration file', default="train")
    parser.add_argument('--device', default='cuda', type=str, help='device', choices=['cuda', 'cpu'])
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    init_distributed_mode(args)
    print(args)
    config = FGVCConfig()
    config.load(args.config)
    cfg = config.cfg
    set_seed(cfg.SEED)
    if args.distributed:
        cfg.DISTRIBUTED = args.distributed
        cfg.GPU = args.gpu
    print(cfg)

    # start task
    if args.task == "train":
        train(cfg)
    else:
        predict(cfg)
    