import numpy as np
import torch
import torch.nn as nn


class CutMix:

    def __init__(self, beta:float=1.0, prob:float=0.5):
        assert beta > 0, "The beta of MixUp Augmentation Should Large than 0"

        self.prob = prob
        self.beta = beta
    
    def aug_data(self, input, target):
        lam = np.random.beta(self.beta, self.beta)
        index = torch.randperm(input.size()[0]).to(input.device)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        input[:, :, bbx1:bbx2, bby1:bby2] = input[index, :, bbx1:bbx2, bby1:bby2]
        target_a, target_b = target, target[index]
        
        return input, target_a, target_b, lam

    def aug_criterion(self, criterion, pred, target_a, target_b, lam):
        return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)
    
    def rand_bbox(self, size, lam):
        
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
