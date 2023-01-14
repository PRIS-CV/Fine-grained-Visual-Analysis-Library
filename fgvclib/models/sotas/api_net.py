import torch
from torch import nn
import numpy as np
from .sota import FGVCSOTA
from fgvclib.criterions.utils import LossItem
from fgvclib.models.sotas import fgvcmodel



@fgvcmodel("APINet")
class APINet(FGVCSOTA):  
    r"""
        Code of Learning Attentive Pairwise Interaction for Fine-Grained Classification (AAAI2020).
        Link: https://github.com/PeiqinZhuang/API-Net 
    """

    def __init__(self, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: dict):
        super().__init__(backbone, encoder, necks, heads, criterions)


    def forward(self, images, targets=None):
        _, _, _, _, conv_out = self.backbone(images)
        pool_out = self.encoder(conv_out).squeeze()

        if self.training:
            intra_pairs, inter_pairs, \
                    intra_labels, inter_labels = self.get_pairs(pool_out, targets)

            #两个相似的特征向量
            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)

            features1_self, features1_other, features2_self, features2_other = self.necks(features1, features2)

            features1_self = tuple([features1_self])
            features1_other = tuple([features1_other])
            features2_self = tuple([features2_self])
            features2_other = tuple([features2_other])
            logit1_self = self.heads(features1_self)
            logit1_other = self.heads(features1_other)
            logit2_self = self.heads(features2_self)
            logit2_other = self.heads(features2_other)

            #训练开始 添加部分
            batch_size = logit1_self.shape[0]
            labels1 = labels1.cuda()
            labels2 = labels2.cuda()

            self_logits = torch.zeros(2*batch_size, 200).cuda()
            other_logits= torch.zeros(2*batch_size, 200).cuda()
            self_logits[:batch_size] = logit1_self
            self_logits[batch_size:] = logit2_self
            other_logits[:batch_size] = logit1_other
            other_logits[batch_size:] = logit2_other
            logits = torch.cat([self_logits, other_logits], dim=0)
            labels = torch.cat([labels1, labels2, labels1, labels2], dim=0)

            losses = list()

            losses.append(LossItem(name="cross_entropy_loss", 
                                    value=self.criterions['cross_entropy_loss']['fn'](logits, labels), 
                                    weight=self.criterions['cross_entropy_loss']['w'])) 
            losses.append(LossItem(name="score_rank_regular_loss", 
                                    value=self.criterions['score_rank_regular_loss']['fn'](self_logits, other_logits, batch_size, labels1, labels2), 
                                    weight=self.criterions['score_rank_regular_loss']['w'])) 
            return logits, losses

        else:
            pool_out = tuple([pool_out])
            logits = self.heads(pool_out)
            return logits

    def pdist(self, vectors):
        distance_matrix = \
            -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
        return distance_matrix

    def get_pairs(self, embeddings, labels):
        distance_matrix = self.pdist(embeddings).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        # short distance
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        # long distance
        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs  = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().cuda()
        intra_pairs = torch.from_numpy(intra_pairs).long().cuda()
        inter_labels = torch.from_numpy(inter_labels).long().cuda()
        inter_pairs = torch.from_numpy(inter_pairs).long().cuda()

        return intra_pairs, inter_pairs, intra_labels, inter_labels
