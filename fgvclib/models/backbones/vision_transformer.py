from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from fgvclib.models.backbones import backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Parameter
from torch.nn.modules.utils import _pair

cfgs = {
    'ViT-B_16',
    'ViT-B_32',
    'ViT-L_16',
    'ViT-L_32',
    'ViT-H_14'
}

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class RelativeCoordPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape

        mask = torch.sum(x, dim=1)
        size = H

        mask = mask.view(N, H * W)
        thresholds = torch.mean(mask, dim=1, keepdim=True)
        binary_mask = (mask > thresholds).float()
        binary_mask = binary_mask.view(N, H, W)

        masked_x = x * binary_mask.view(N, 1, H, W)
        masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)

        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()

        basic_label = torch.from_numpy(self.build_basic_label(size)).float()
        # Build Label
        label = basic_label.cuda()
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 1, 2)
        relative_coord = label - basic_anchor
        relative_coord = relative_coord / size
        relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)

        binary_relative_mask = binary_mask.view(N, H * W)
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask

        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)

        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)

        position_weight = torch.mean(masked_x, dim=-1)
        position_weight = position_weight.unsqueeze(2)
        position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))

        return relative_coord_total, basic_anchor, position_weight, reduced_x_max_index

    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, cfg: dict, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = cfg["patch_size"]
        if cfg['split'] == 'non-overlap':
            n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=cfg['hidden_size'],
                                           kernel_size=patch_size,
                                           stride=patch_size)
        elif cfg['split'] == 'overlap':
            n_patches = ((img_size[0] - patch_size) // cfg['slide_step'] + 1) * (
                    (img_size[1] - patch_size) // cfg['slide_step'] + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=cfg['hidden_size'],
                                           kernel_size=patch_size,
                                           stride=(cfg['slide_step'], cfg['slide_step']))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, cfg['hidden_size']))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg['hidden_size']))

        self.dropout = Dropout(cfg['dropout_rate'])

        self.struc_token = nn.Parameter(torch.zeros(1, 1, cfg['hidden_size']))
        self.relative_coord_predictor = RelativeCoordPredictor()
        self.struct_head = nn.Sequential(
            nn.BatchNorm1d(37 * 37 * 2 + 2),
            Linear(37 * 37 * 2 + 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, cfg['hidden_size']),
        )
        self.act_fn = ACT2FN["relu"]

    def get_attn(self, x, y):
        attn = F.normalize(x) * F.normalize(y)
        attn = torch.sum(attn, -1)

        H = attn.size(1) ** 0.5
        H = int(H)

        attn = attn.contiguous().view(attn.size(0), H, H)
        attn = attn.unsqueeze(dim=1)

        B, C, H, W = attn.shape
        structure_info = self.relative_coord_predictor(attn)
        structure_info = self.struct_head(structure_info)
        structure_info = self.act_fn(structure_info)
        structure_info = structure_info.unsqueeze(1)

        return structure_info

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        struc_tokens = self.struc_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, struc_tokens


def _vit(cfg, model_name="ViT-B_16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    return Embeddings(cfg, img_size=cfg['image_size'])


@backbone("vision_transformer")
def vision_transformer(cfg):
    return _vit(cfg, "ViT-B_16")
