# Copyright (c) PRIS-CV. All rights reserved.
import torch.nn as nn
from fgvclib.models.utils import BasicConv
from fgvclib.models.necks import neck

class MultiScaleConv(nn.Module):

    def __init__(self, scale_num=3, in_dim=[512, 512, 512], hid_dim=[512, 512, 512], out_dim=[512, 512, 512]):
        super(MultiScaleConv, self).__init__()
        assert scale_num == len(in_dim), f'The length of input dimension {len(in_dim)} should be aligned to the scale number {scale_num}.'
        assert scale_num == len(hid_dim), f'The length of hide dimension {len(hid_dim)} should be aligned to the scale number {scale_num}.'
        assert scale_num == len(out_dim), f'The length of output dimension {len(out_dim)} should be aligned to the scale number {scale_num}.'
        
        self.scale_num = scale_num
        self.conv_blocks = nn.ModuleList()
        for s in range(scale_num):
            self.conv_blocks.append(nn.Sequential(
            BasicConv(in_dim[s], hid_dim[s], kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(hid_dim[s], out_dim[s], kernel_size=3, stride=1, padding=1, relu=True)
        ))
    
    def forward(self, inputs):
        
        assert self.scale_num == len(inputs), 'The length of input {len(inputs)} should be aligned to the scale number {self.scale_num}.'
        
        outputs = [self.conv_blocks[s](inputs[s]) for s in range(self.scale_num)]
        
        return tuple(outputs)

@neck("multi_scale_conv")        
def multi_scale_conv(cfg: dict) -> MultiScaleConv:
    
    if cfg is not None:
        
        assert "scale_num" in cfg.keys()
        assert isinstance(cfg["scale_num"], int)
        assert "in_dim" in cfg.keys()
        assert isinstance(cfg["in_dim"], list) 
        assert "hid_dim" in cfg.keys()
        assert isinstance(cfg["hid_dim"], list) 
        assert "out_dim" in cfg.keys()
        assert isinstance(cfg["out_dim"], list)

        return MultiScaleConv(scale_num=cfg["scale_num"], in_dim=cfg["in_dim"], hid_dim=cfg["hid_dim"], out_dim=cfg["out_dim"])
    
    return MultiScaleConv()

            