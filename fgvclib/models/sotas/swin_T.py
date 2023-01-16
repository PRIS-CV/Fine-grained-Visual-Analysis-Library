import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode

from .sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel


@fgvcmodel("Swin_T")
class Swin_T(FGVCSOTA):
    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)

        ### get hidden feartues size
        img_size = self.args["img_size"]
        num_classes = cfg.CLASS_NUM
        rand_in = torch.randn(1, 3, img_size, img_size)
        outs = self.backbone(rand_in)

        ### = = = = = FPN = = = = =
        self.fpn = self.encoder
        fpn_size = 1536
        self.build_fpn_classifier(outs, fpn_size, num_classes)

        ### = = = = = Selector = = = = =
        self.selector = self.necks

        ### = = = = = Combiner = = = = =
        self.combiner = self.heads

        ### just original backbone
        if not self.fpn and (not self.combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size.size(-1)
                elif len(fs_size) == 4:
                    out_size = fs_size.size(1)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1)
            )
            self.add_module("fpn_classifier_" + name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def fpn_predict(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H * W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_" + name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous()  # transpose

    def forward(self, x: torch.Tensor):

        logits = {}
        x = self.forward_backbone(x)

        if self.fpn:
            x = self.fpn(x)
            self.fpn_predict(x, logits)

        if self.selector:
            selects = self.selector(x, logits)

        if self.combiner:
            comb_outs = self.combiner(selects)
            logits['comb_outs'] = comb_outs
            return logits

        if self.selector or self.fpn:
            return logits

        ### original backbone (only predict final selected layer)
        for name in x:
            hs = x[name]

            if len(hs.size()) == 4:
                hs = F.adaptive_avg_pool2d(hs, (1, 1))
                hs = hs.flatten(1)
            else:
                hs = hs.mean(1)
            out = self.classifier(hs)
            logits['ori_out'] = logits

            return logits
