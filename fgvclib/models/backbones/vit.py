import logging
import torch
import torch.nn as nn

from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair
from fgvclib.models.backbones import backbone

# official pretrain weights
model_urls = {
    'ViT-B_16': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
    'ViT-B_32': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz',
    'ViT-L_16': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz',
    'ViT-L_32': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz',
    'ViT-H_14': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz'
}

cfgs = {
    'ViT-B_16',
    'ViT-B_32',
    'ViT-L_16',
    'ViT-L_32',
    'ViT-H_14'
}

logger = logging.getLogger(__name__)


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

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def _vit(cfg, model_name="ViT-B_16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    return Embeddings(cfg, img_size=cfg['image_size'])


@backbone("vit16")
def vit16(cfg):
    return _vit(cfg, "ViT-B_16")
