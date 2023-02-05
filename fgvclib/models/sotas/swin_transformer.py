import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode

from .sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel
from fgvclib.criterions.utils import LossItem
from fgvclib.models.encoders.fpn import FPN
from fgvclib.models.necks.weakly_selector import WeaklySelector

@fgvcmodel("SwinTransformer")
class SwinTransformer(FGVCSOTA):
    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)

        ### get hidden feartues size
        
        
        self.num_classes = cfg.CLASS_NUM
        self.use_fpn = self.args["use_fpn"]
        self.lambda_s = self.args["lambda_s"]
        self.lambda_n = self.args["lambda_n"]
        self.lambda_b = self.args["lambda_b"]
        self.lambda_c = self.args["lambda_c"]
        self.use_combiner = self.args["use_combiner"]
        self.update_freq = self.args["update_freq"]
        self.use_selection = self.args["use_selection"]
        num_select = self.args["num_select"]
        
        input_size = self.args["img_size"]
        rand_in = torch.randn(1, 3, input_size, input_size)
        backbone_outs = self.backbone(rand_in)
        
        if self.use_fpn:
            
            fpn_size = self.args["fpn_size"]
            self.encoder = FPN(inputs=backbone_outs, fpn_size=fpn_size, proj_type="Linear", upsample_type="Conv")
            fpn_outs = self.encoder(backbone_outs)
        else:
            fpn_outs = backbone_outs
            fpn_size = None
        
        
        self.necks = WeaklySelector(inputs=fpn_outs, num_classes=self.num_classes, num_select=num_select, fpn_size=fpn_size)

        ### = = = = = FPN = = = = =
        self.fpn = self.encoder
        
        self.build_fpn_classifier(backbone_outs, fpn_size, self.num_classes)

        ### = = = = = Selector = = = = =
        self.selector = self.necks

        ### = = = = = Combiner = = = = =
        self.combiner = self.heads

        ### just original backbone
        if not self.fpn and (not self.combiner):
            for name in backbone_outs:
                fs_size = backbone_outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size.size(-1)
                elif len(fs_size) == 4:
                    out_size = fs_size.size(1)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, self.num_classes)

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

    def infer(self, x: torch.Tensor):

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

    def forward(self, x, target=None):
        
        logits = self.infer(x)

        if not self.training:
            return logits
        else:
        
            losses = list()
            batch_size = x.shape[0]
            device = x.device
            for name in logits:

                if "select_" in name:
                    if not self.use_selection:
                        raise ValueError("Selector not use here.")
                    if self.lambda_s != 0:
                        S = logits[name].size(1)
                        logit = logits[name].view(-1, self.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit.float(), target.unsqueeze(1).repeat(1, S).flatten(0))
                        losses.append(LossItem(name="loss_s", value=loss_s, weight=self.lambda_s))

                elif "drop_" in name:
                    if not self.use_selection:
                        raise ValueError("Selector not use here.")

                    if self.lambda_n != 0:
                        S = logits[name].size(1)
                        logit = logits[name].view(-1, self.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = (torch.zeros([batch_size * S, self.num_classes]) - 1).to(device)
                        loss_n = nn.MSELoss()(n_preds.float(), labels_0)
                        losses.append(LossItem(name="loss_n", value=loss_n, weight=self.lambda_n))
                    

                elif "layer" in name:
                    if not self.use_fpn:
                        raise ValueError("FPN not use here.")
                    if self.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(logits[name].mean(1).float(), target)
                        losses.append(LossItem(name="loss_b", value=loss_b, weight=self.lambda_b))

                elif "comb_outs" in name:
                    if not self.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if self.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(logits[name].float(), target)
                        losses.append(LossItem(name="loss_c", value=loss_c, weight=self.lambda_c))

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(logits[name].float(), target)
                    losses.append(LossItem(name="loss_ori", value=loss_ori, weight=1.0))

            return logits, losses
