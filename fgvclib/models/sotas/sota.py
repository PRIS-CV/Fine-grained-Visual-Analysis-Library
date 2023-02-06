from yacs.config import CfgNode
import torch.nn as nn
from thop import profile

from fgvclib.configs.utils import turn_list_to_dict as tltd


class FGVCSOTA(nn.Module):
    
    def __init__(self, cfg:CfgNode, backbone:nn.Module, encoder:nn.Module, necks:nn.Module, heads:nn.Module, criterions:nn.Module):
        super(FGVCSOTA, self).__init__()
        
        self.cfg = cfg
        self.backbone = backbone
        self.necks = necks
        self.encoder = encoder
        self.heads = heads
        self.criterions = criterions
        self.args = tltd(cfg.ARGS)

    def get_structure(self):
        ss = f"\n{'=' * 30}\n" + f"\nThe Structure of {self.__class__.__name__}:\n" + f"\n{'=' * 30}\n\n"
        for s in ['backbone', 'encoder', 'necks', 'heads']:
            m = getattr(self, s)
            ss += f"{s}: {m.__class__.__name__}\n"
        ss += f"\n{'=' * 30}\n"
        return ss

    def get_statistics(self, input):
        flops, params = profile(self, inputs=(input, ))
        print(f"Flops: {flops}")
        print(f"Params: {params}")
        