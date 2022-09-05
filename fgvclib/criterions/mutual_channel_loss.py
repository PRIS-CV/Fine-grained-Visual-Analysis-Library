import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .utils import LossItem

class MutualChannelLoss(nn.Module):
    
    def __init__(self, height=None, cnum=None, div_weight=None, dis_weight=None):
        super(MutualChannelLoss, self).__init__()
        
        self.height = height
        self.cnum = cnum
        self.criterion = nn.CrossEntropyLoss()
        self.div_weight = div_weight
        self.dis_weight = dis_weight
        self.maxpool2d = _MaxPool2d(kernel_size=(1, self.cnum), stride=(1, self.cnum))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(self.height, self.height))
    
    def forward(self, x, targets, class_num):
        div_loss = self.diversity_loss(x)
        dis_loss = self.discriminality_loss(x, targets, class_num)

        losses = [
            LossItem(name='mc_div_loss', value=div_loss, weight=self.div_weight),
            LossItem(name='mc_dis_loss', value=dis_loss, weight=self.dis_weight)
        ]
        return losses
    
    def cwa(self, nb_batch, channels, class_num, device):     # channel dropout

        mask_per_class = [1] * 2 + [0] *  1
        mask_per_sample = []
        for i in range(class_num):
            random.shuffle(mask_per_class)
            mask_per_sample += mask_per_class
        mask_all_batch = [mask_per_sample for _ in range(nb_batch)]  
        mask_all_batch = np.array(mask_all_batch).astype("float32")
        mask_all_batch = mask_all_batch.reshape(nb_batch, 200 * channels, 1, 1)
        mask_all_batch = torch.from_numpy(mask_all_batch)
        mask_all_batch = mask_all_batch.to(device)
        mask_all_batch = torch.autograd.Variable(mask_all_batch)
        return mask_all_batch

    def discriminality_loss(self, x, targets, class_num):
        
        channel_masks = self.cwa(x.size(0), self.cnum, class_num, x.device)
        
        x = x * channel_masks
        x = self.maxpool2d(x) 
        x = self.avgpool2d(x)
        x = x.flatten(start_dim=1)

        loss = self.criterion(x, targets)  # Discriminality Component

        return loss
    
    def diversity_loss(self, x):
        s = x.size()
        x = x.flatten(start_dim=2)
        x = F.softmax(x, dim=2)
        x = x.reshape(s)
        x = self.maxpool2d(x)  
        x = x.flatten(start_dim=2)
        loss = 1.0 - 1.0 * torch.mean(torch.sum(x, 2)) / self.cnum
        return loss

class _MaxPool2d(nn.Module):


    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)

        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class _AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()

        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'



def mutual_channel_loss(cfg=None):
    assert 'height' in cfg.keys(),           'height must exist in parameters'
    assert 'cnum' in cfg.keys(),             'cnum must exist in parameters'
    assert 'div_weight' in cfg.keys(),       'div_weight must exist in parameters'
    assert 'dis_weight' in cfg.keys(),       'dis_weight must exist in parameters'
    return MutualChannelLoss(height=cfg['height'], cnum=cfg['cnum'], div_weight=cfg['div_weight'], dis_weight=cfg['dis_weight'])