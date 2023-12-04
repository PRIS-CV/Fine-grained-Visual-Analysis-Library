from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import encoder
from ..backbones.vision_transformer import Embeddings
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Parameter

from fgvclib.models.backbones.vision_transformer import RelativeCoordPredictor
from fgvclib.models.heads.mlp import mlp

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super(Attention, self).__init__()
        self.num_attention_heads = cfg['num_heads']
        self.attention_head_size = int(cfg['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(cfg['hidden_size'], self.all_head_size)
        self.key = Linear(cfg['hidden_size'], self.all_head_size)
        self.value = Linear(cfg['hidden_size'], self.all_head_size)

        self.out = Linear(cfg['hidden_size'], cfg['hidden_size'])
        self.attn_dropout = Dropout(cfg['attention_dropout_rate'])
        self.proj_dropout = Dropout(cfg['attention_dropout_rate'])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super(Block, self).__init__()
        self.hidden_size = cfg['hidden_size']
        self.attention_norm = LayerNorm(cfg['hidden_size'], eps=1e-6)
        self.ffn_norm = LayerNorm(cfg['hidden_size'], eps=1e-6)
        self.ffn = mlp(cfg)
        self.attn = Attention(cfg)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)

        last_map = last_map[:, :, 0, 1:]
        max_value, max_inx = last_map.max(2)

        B, C = last_map.size(0), last_map.size(1)
        patch_num = last_map.size(-1)

        H = patch_num ** 0.5
        H = int(H)
        attention_map = last_map.view(B, C, H, H)

        return last_map, max_inx, max_value, attention_map


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        weight = self.weight.float()
        support = torch.matmul(input, weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x


class Part_Structure(nn.Module):
    def __init__(self, cfg: dict):
        super(Part_Structure, self).__init__()
        self.fc1 = Linear(37 * 37 * 2 + 2, cfg['hidden_size'])
        self.act_fn = ACT2FN["relu"]
        self.dropout = Dropout(cfg["dropout_rate"])
        self.relative_coord_predictor = RelativeCoordPredictor()

        self.struct_head = nn.Sequential(
            nn.BatchNorm1d(37 * 37 * 2 + 2),
            Linear(37 * 37 * 2 + 2, cfg['hidden_size']),
        )

        self.struct_head_new = nn.Sequential(
            nn.BatchNorm1d(cfg['hidden_size'] * 2),
            Linear(cfg['hidden_size'] * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, cfg['hidden_size']),
        )

        self.gcn = GCN(2, 512, cfg['hidden_size'], dropout=0.1)

    def forward(self, hidden_states, part_inx, part_value, attention_map, struc_tokens):
        B, C, H, W = attention_map.shape
        structure_info, basic_anchor, position_weight, reduced_x_max_index = self.relative_coord_predictor(
            attention_map)
        structure_info = self.gcn(structure_info, position_weight)

        for i in range(B):
            index = int(basic_anchor[i, 0] * H + basic_anchor[i, 1])

            hidden_states[i, 0] = hidden_states[i, 0] + structure_info[i, index, :]

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(cfg["num_layers"] - 1):
            layer = Block(cfg)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention()
        self.part_layer = Block(cfg)
        self.part_norm = LayerNorm(cfg['hidden_size'], eps=1e-6)
        self.part_structure = Part_Structure(cfg)

    def forward(self, hidden_states, struc_tokens):
        attn_weights = []
        hid_ori = []
        structure = []

        for i, layer in enumerate(self.layer):
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)

            if i > 8:
                temp_weight = []
                temp_weight.append(weights)
                _, part_inx, part_value, a_map = self.part_select(temp_weight)
                hidden_states = self.part_structure(hidden_states, part_inx, part_value, a_map, struc_tokens)

                hid_ori.append(self.part_norm(hidden_states[:, 0]))

        part_states, part_weights = self.part_layer(hidden_states)
        attn_weights.append(part_weights)
        temp_weight = []
        temp_weight.append(part_weights)
        _, part_inx, part_value, a_map = self.part_select(temp_weight)
        part_states = self.part_structure(part_states, part_inx, part_value, a_map, struc_tokens)
        part_encoded = self.part_norm(part_states)

        return part_encoded, hid_ori


class Transformer(nn.Module):
    def __init__(self, cfg: dict, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(cfg, img_size=img_size)
        self.encoder = Encoder(cfg)

    def forward(self, input_ids):
        embedding_output, struc_tokens = self.embeddings(input_ids)
        part_encoded, hid = self.encoder(embedding_output, struc_tokens)
        # print(hid.shape)
        return part_encoded, hid


@encoder("sim_trans_encoder")
def sim_trans_encoder(cfg: dict):
    return Transformer(cfg, img_size=cfg['img_size'])
