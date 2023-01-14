# Copyright (c) PRIS-CV. All rights reserved.
import torch
import torch.nn as nn
from fgvclib.models.necks import neck

class PairwiseInter(nn.Module):

    def __init__(self, in_dim=4096, hid_dim=512, out_dim=2048):
        super(PairwiseInter, self).__init__()
        self.map1 = nn.Linear(in_dim, hid_dim)
        self.map2 = nn.Linear(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
    
    def forward(self, features1, features2):

        #通过MLP生成一个互向量
        mutual_features = torch.cat([features1, features2], dim=1)
        map1_out = self.map1(mutual_features)
        map2_out = self.drop(map1_out)
        map2_out = self.map2(map2_out)

        #门向量生成
        gate1 = torch.mul(map2_out, features1)
        gate1 = self.sigmoid(gate1)

        gate2 = torch.mul(map2_out, features2)
        gate2 = self.sigmoid(gate2)

        #成对交互
        features1_self = torch.mul(gate1, features1) + features1
        features1_other = torch.mul(gate2, features1) + features1

        features2_self = torch.mul(gate2, features2) + features2
        features2_other = torch.mul(gate1, features2) + features2
        
        return features1_self, features1_other, features2_self, features2_other 

@neck("pairwise_interaction")
def pairwise_interaction(cfg: dict) -> PairwiseInter:
    
    if cfg is not None:
        
        assert "in_dim" in cfg.keys()
        assert isinstance(cfg["in_dim"], int) 
        assert "hid_dim" in cfg.keys()
        assert isinstance(cfg["hid_dim"], int) 
        assert "out_dim" in cfg.keys()
        assert isinstance(cfg["out_dim"], int)

        return PairwiseInter(in_dim=cfg["in_dim"], hid_dim=cfg["hid_dim"], out_dim=cfg["out_dim"])
    
    return PairwiseInter()

            