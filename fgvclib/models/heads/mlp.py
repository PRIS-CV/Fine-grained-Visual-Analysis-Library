from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from fgvclib.models.heads import head


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


@head("mlp")
def mlp(cfg: dict) -> Mlp:
    assert 'hidden_size' in cfg.keys()
    assert 'mlp_dim' in cfg.keys()
    assert 'dropout_rate' in cfg.keys()
    return Mlp(hidden_size=cfg['hidden_size'], mlp_dim=cfg['mlp_dim'], dropout_rate=cfg['dropout_rate'])
