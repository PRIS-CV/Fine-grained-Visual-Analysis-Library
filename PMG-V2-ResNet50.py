from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.optim as optim
from torchvision import transforms

from fgvclib.models.sotas import PMG_V2_ResNet50
from fgvclib.apis import evaluate_model, update_model
from fgvclib.datasets import Dataset_AnnoFolder
from fgvclib.utils.lr_schedules import cosine_anneal_schedule


def main():
    # Args
    epoch_num = 200
    batch_size = 4
    start_epoch = 0
    resume = None
    use_cuda = torch.cuda.is_available()
    
    # Transform
    transform_train = transforms.Compose([
        transforms.Scale((600, 600)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Scale((600, 600)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Data Loader
    train_set = Dataset_AnnoFolder(root='/data/duruoyi/fg2/Birds2/train', transform=transform_train, positive=1)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_set = Dataset_AnnoFolder(root='/data/duruoyi/fg2/Birds2/test', transform=transform_test, positive=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    if resume != None:
        model = torch.load(resume)
    else:
        model = PMG_V2_ResNet50()
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model)
        
    
    # Optimizer
    try:
        optimizer = optim.SGD([
            {'params': model.backbone.parameters(), 'lr': 0.0005},
            {'params': model.necks.parameters(), 'lr': 0.005},
            {'params': model.heads.parameters(), 'lr': 0.005},
        ], momentum=0.9, weight_decay=5e-4)
    except Exception:
        optimizer = optim.SGD([
        {'params': model.module.backbone.parameters(), 'lr': 0.0005},
        {'params': model.module.necks.parameters(), 'lr': 0.005},
        {'params': model.module.heads.parameters(), 'lr': 0.005},
    ], momentum=0.9, weight_decay=5e-4)

    # Train and Evaluate
    for epoch in range(start_epoch, epoch_num):
        cosine_anneal_schedule(optimizer, epoch, epoch_num)
        update_model(model, train_loader, optimizer, strategy="progressive_updating_consistency_constraint", use_cuda=True)
        acc = evaluate_model(model, test_loader, metrics=["top1-accuracy"], use_cuda=True)
        
if __name__=='__main__':
    main()