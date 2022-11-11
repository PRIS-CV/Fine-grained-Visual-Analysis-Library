# Copyright (c) PRIS-CV. All rights reserved.
import torch
import torch.nn as nn

class GlobalMaxPooling(nn.Module):

    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, inputs, concat=False):
        if isinstance(inputs, tuple):
            outs = [self.gmp(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
            if concat:
                out_concat = torch.cat(outs, 1)
                outs.append(out_concat)
            outs = tuple(outs)
        elif isinstance(inputs, torch.Tensor):
            outs = self.gmp(inputs)
            outs = outs.view(inputs.size(0), -1)
            if concat:    
                raise TypeError('Inputs of GlobalMaxPooling with concat=True should be tuple')
        else:
            raise TypeError('GlobalMaxPooling inputs should be tuple or torch.tensor')
        
        return outs

class GlobalAvgPooling(nn.Module):

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs, concat=False):
        if isinstance(inputs, tuple):
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
            if concat:
                out_concat = torch.cat(outs, 1)
                outs.append(out_concat)
            outs = tuple(outs)
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
            if concat:    
                raise TypeError('Inputs of GlobalAvgPooling with concat=True should be tuple')
        else:
            raise TypeError('GlobalAvgPooling inputs should be tuple or torch.tensor')
        
        return outs

def global_avg_pooling(cfg:dict=None):
    return GlobalAvgPooling()

def global_max_pooling(cfg:dict=None):
    return GlobalMaxPooling()

def max_pooling_2d(cfg:dict):
    assert 'kernel_size' in cfg.keys()
    assert isinstance(cfg['kernel_size'], int) 
    assert 'stride' in cfg.keys()
    assert isinstance(cfg['stride'], int)
    return nn.MaxPool2d(kernel_size=cfg['kernel_size'], stride=cfg['stride'])

def max_pooling_2d(cfg:dict):
    assert 'kernel_size' in cfg.keys()
    assert isinstance(cfg['kernel_size'], int) 
    assert 'stride' in cfg.keys()
    assert isinstance(cfg['stride'], int)
    return nn.AvgPool2d(kernel_size=cfg['kernel_size'], stride=cfg['stride'])
            