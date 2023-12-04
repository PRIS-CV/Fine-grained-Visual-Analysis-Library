import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from fgvclib.models.utils.tree_elements import *
import matplotlib.pyplot as plt
import os

from fgvclib.models.heads import head

class DTree(nn.Module):
    mean = 0.5
    std = 0.1

    def __init__(self, num_classes, tree_depth, embed_dim, proto_size):
        super(DTree, self).__init__()

        self.proto_size = proto_size
        self.num_classes = num_classes
        self._parents = dict()
        self._root = self._init_tree(num_classes, tree_depth, proto_size)
        self._set_parents()
        self.img_size = 224 // 16

        self._out_map = {n.index: i for i, n in zip(range(2 ** (tree_depth) - 1), self.branches)}
        self._maxims = {n.index: float('-inf') for n in self.branches}
        self.num_prototypes = self.num_branches
        self.num_leaves = len(self.leaves)
        self._leaf_map = {n.index: i for i, n in zip(range(self.num_leaves), self.leaves)}
        self.proto_dim = embed_dim // 4

        prototype_shape = [self.num_prototypes, self.proto_dim] + self.proto_size
        self.add_on = nn.Sequential(
            nn.Linear(embed_dim, self.proto_dim, bias=False),
            nn.Sigmoid()
        )
        self.attrib_pvec = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)
        self._init_param()

    def forward(self, logits, patches, pool_map, attn_weights, **kwargs):
        patches = self.add_on(patches)
        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['attn_weights'] = attn_weights

        out, attr = self._root.forward(logits, patches, **kwargs)

        return out

    def hard_forward(self, logits, patches, pool_map, attn_weights, **kwargs):
        patches = self.add_on(patches)
        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['attn_weights'] = attn_weights

        out, attr = self._root.hard_forward(logits, patches, **kwargs)

        return out

    def explain_internal(self, logits, patches, x_np, pool_map, attn_weights, **kwargs):
        patches = self.add_on(patches)

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['maxims'] = dict(self._maxims)
        kwargs['img_size'] = self.img_size

        out, attr = self._root.explain_internal(logits, patches, x_np, 0, **kwargs)
        return out

    def explain(self, logits, patches, x_np, y, pool_map, prefix: str, **kwargs):
        patches = self.add_on(patches)

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['img_size'] = self.img_size

        if not os.path.exists(os.path.join('heatmap', prefix)):
            os.makedirs(os.path.join('heatmap', prefix))

        plt.imsave(fname=os.path.join('heatmap', prefix, 'input_image.png'),
                   arr=x_np, vmin=0.0, vmax=1.0)

        r_node_id = 0
        l_sim = None
        r_sim = None
        out, attr = self._root.explain(logits, patches, l_sim, r_sim, x_np, y, prefix, r_node_id, pool_map,
                                       **kwargs)

        return out

    def get_min_by_ind(self, left_distance, right_distance):
        B, Br, W, H = left_distance.shape

        relative_distance = left_distance / (left_distance + right_distance)
        relative_distance = relative_distance.view(B, Br, -1)
        _, min_dist_idx = relative_distance.min(-1)
        min_left_distance = left_distance.view(B, Br, -1).gather(-1, min_dist_idx.unsqueeze(-1))
        return min_left_distance

    def _init_tree(self, num_classes, tree_depth, proto_size):
        def _init_tree_recursive(i, d):
            if d == tree_depth:
                return Leaf(i, num_classes)
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                right = _init_tree_recursive(i + left.size + 1, d + 1)
                return Branch(i, left, right, proto_size)

        return _init_tree_recursive(0, 0)

    def _init_param(self):
        def init_weights_xavier(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

        with torch.no_grad():
            torch.nn.init.normal_(self.attrib_pvec, mean=self.mean, std=self.std)
            self.add_on.apply(init_weights_xavier)

    def _set_parents(self):
        def _set_parents_recursively(node):
            if isinstance(node, Branch):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return
            raise Exception('Unrecognized node type!')

        self._parents.clear()
        self._parents[self._root] = None
        _set_parents_recursively(self._root)

    @property
    def root(self):
        return self._root

    @property
    def leaves_require_grad(self) -> bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self.backbone.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self.backbone.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    @property
    def depth(self) -> int:
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) -> int:
        return self._root.size

    @property
    def nodes(self) -> set:
        return self._root.nodes

    @property
    def nodes_by_index(self) -> dict:
        return self._root.nodes_by_index

    @property
    def node_depths(self) -> dict:

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}

        return _assign_depths(self._root, 0)

    @property
    def branches(self) -> set:
        return self._root.branches

    @property
    def leaves(self) -> set:
        return self._root.leaves

    @property
    def num_branches(self) -> int:
        return self._root.num_branches

@head("dtree")
def dtree(cfg: dict, class_num: int) -> DTree:
    assert 'depth' in cfg.keys()
    assert 'proto_size' in cfg.keys()
    
    return DTree(num_classes=class_num, tree_depth=cfg['depth'], embed_dim=1024, proto_size=cfg['proto_size'])  