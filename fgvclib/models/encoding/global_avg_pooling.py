# Copyright (c) PRIS-CV. All rights reserved.
import torch
import torch.nn as nn

class GlobalAvgPooling(nn.Module):
    """Global average pooling encoding.

    Note that ...

    Args:
        
    """

    def __init__(self, cfg):
        super(GlobalAvgPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cfg = cfg

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
            