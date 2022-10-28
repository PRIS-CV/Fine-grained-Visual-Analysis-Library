import numpy as np
import torch
import torch.nn as nn


class MixUp(nn.Module):
    r"""The mixup data augmentation 
    """

    def __init__(self, beta:float=1.0, prob:float=0.5):

        self.prob = prob
        assert beta > 0, "The beta of MixUp Augmentation Should Large than 0"
        self.beta = beta

    def aug_data(self, input, target):
        
        lam = np.random.beta(self.beta, self.beta)
        index = torch.randperm(input.size()[0]).to(input.device)
        input = lam * input + (1 - lam) * input[index, :]
        target_a, target_b = target, target[index]
        return input, target_a, target_b, lam

    def aug_criterion(criterion, pred, target_a, target_b, lam):
        return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)
