"""
WS-DAN models
Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode

from .sota import FGVCSOTA
from fgvclib.criterions.utils import LossItem
from fgvclib.models.sotas import fgvcmodel

import random

__all__ = ['WSDAN_CAL']
EPSILON = 1e-6

def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.interpolate(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.interpolate(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

@fgvcmodel("WSDAN_CAL")
class WSDAN_CAL(FGVCSOTA):
    r"""
        Code of Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification (ICCV2021).
        Link: https://github.com/raoyongming/CAL 
    """

    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
        
        self.out_channels = 2048
        self.num_classes = cfg.CLASS_NUM
        self.M = 32
        self.net = 'resnet101'
        self.register_buffer('feature_center', torch.zeros(self.num_classes, self.M * self.out_channels))   # 32 * 2048


    def infer(self, x):
        batch_size = x.size(0)

        _, _, _, _, feature_maps = self.backbone(x)
        
        attention_maps = self.necks(feature_maps)
        
        feature_matrix, feature_matrix_hat = self.encoder(feature_maps, attention_maps)

        # Classification
        p = self.heads(tuple([feature_matrix * 100.]))

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
        
        return p, p - self.heads(tuple([feature_matrix_hat * 100.])), feature_matrix, attention_map
    
    def forward(self, x, targets=None):
        if self.training:
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.infer(x)

            # Update Feature Center
            beta = 5e-2 
            feature_center_batch = F.normalize(self.feature_center[targets], dim=-1)
            self.feature_center[targets] += beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
            
            aug_images = torch.cat([crop_images, drop_images], dim=0)
            y_aug = torch.cat([targets, targets], dim=0)

            # crop images forward
            y_pred_aug, y_pred_aux_aug, _, _ = self.infer(aug_images)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
            y_aux = torch.cat([targets, y_aug], dim=0)

            # loss
            losses = list()
            losses.append(LossItem(name="cross_entropy_loss", 
                                   value=self.criterions['cross_entropy_loss']['fn'](y_pred_raw, targets), 
                                   weight=self.criterions['cross_entropy_loss']['w']/3.)) 
            losses.append(LossItem(name="cross_entropy_loss", 
                                   value=self.criterions['cross_entropy_loss']['fn'](y_pred_aux, y_aux), 
                                   weight=self.criterions['cross_entropy_loss']['w']* 3. / 3.)) 
            losses.append(LossItem(name="cross_entropy_loss", 
                                   value=self.criterions['cross_entropy_loss']['fn'](y_pred_aug, y_aug), 
                                   weight=self.criterions['cross_entropy_loss']['w']* 2. / 3.)) 
            losses.append(LossItem(name="center_loss", 
                                   value=self.criterions['center_loss']['fn'](feature_matrix, feature_center_batch), 
                                   weight=self.criterions['center_loss']['w'])) 
            return y_pred_raw, losses
        
        else:
            
            y_pred_raw, y_pred_aux, _, attention_map = self.infer(x)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = self.infer(crop_images3)
            
            y_pred = (y_pred_raw + y_pred_crop3) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.
            return y_pred


    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN_CAL, self).load_state_dict(model_dict)

    def infer_aux(self, x):
        x_m = torch.flip(x, [3])

        y_pred_raw, y_pred_aux_raw, _, attention_map = self.infer(x)
        y_pred_raw_m, y_pred_aux_raw_m, _, attention_map_m = self.infer(x_m)
        crop_images = batch_augment(x, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
        y_pred_crop, y_pred_aux_crop, _, _ = self.infer(crop_images)

        crop_images2 = batch_augment(x, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
        y_pred_crop2, y_pred_aux_crop2, _, _ = self.infer(crop_images2)

        crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop3, y_pred_aux_crop3, _, _ = self.infer(crop_images3)

        crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
        y_pred_crop_m, y_pred_aux_crop_m, _, _ = self.infer(crop_images_m)

        crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
        y_pred_crop_m2, y_pred_aux_crop_m2, _, _ = self.infer(crop_images_m2)

        crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop_m3, y_pred_aux_crop_m3, _, _ = self.infer(crop_images_m3)

        y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
        y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
        y_pred = (y_pred + y_pred_m) / 2.

        y_pred_aux = (y_pred_aux_raw + y_pred_aux_crop + y_pred_aux_crop2 + y_pred_aux_crop3) / 4.
        y_pred_aux_m = (y_pred_aux_raw_m + y_pred_aux_crop_m + y_pred_aux_crop_m2 + y_pred_aux_crop_m3) / 4.
        y_pred_aux = (y_pred_aux + y_pred_aux_m) / 2.

        return y_pred_aux
