
import torch
from torch import nn
from yacs.config import CfgNode

from fgvclib.configs.utils import turn_list_to_dict as tltd
from fgvclib.models.sotas.sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel
from fgvclib.criterions import LossItem
from fgvclib.transforms import MixUpCutMix

@fgvcmodel("ViT_NeT")
class ViT_NeT(FGVCSOTA):
    r"""
        Code of "ViT-NeT: Interpretable Vision Transformers with Neural Tree Decoder".
        Link: https://github.com/jumpsnack/ViT-NeT
    """

    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
        num_classes = cfg.CLASS_NUM
        self.mixup_fn = MixUpCutMix(cfg=tltd(cfg.ARGS), num_classes= num_classes)

    def forward(self, x, targets=None):
        if self.training:
            x, targets = self.mixup_fn(x,targets)
        losses = list()
        logits, patches = self.backbone(x)
        out = self.heads(logits, patches, None, None)

        if self.training:
            losses.append(LossItem(name="soft_target_cross_entropy_loss", 
                                   value=self.criterions['soft_target_cross_entropy_loss']['fn'](torch.log(out + 1e-12), targets), 
                                   weight=self.criterions['soft_target_cross_entropy_loss']['w'])) 

            return out, losses

        return out

    def hard_forward(self, x):
        logits, patches = self.backbone(x)
        out = self.heads.hard_forward(logits, patches, None, None)
        return out

    def explain(self, x, y, prefix):
        x_np = np.clip(((x + 1) / 2).permute(0, 2, 3, 1)[0].cpu().numpy(), 0., 1.)
        logits, patches = self.backbone(x)
        self.heads.explain(logits, patches, x_np, y, None, prefix)

    def explain_internal(self, x):
        x_np = np.clip(((x + 1) / 2).permute(0, 2, 3, 1)[0].cpu().numpy(), 0., 1.)
        logits, patches = self.backbone(x)
        out = self.heads.explain_internal(logits, patches, x_np, None, None)
        return out
    