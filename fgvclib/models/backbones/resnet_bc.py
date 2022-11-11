import torch.nn as nn
from  torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch


from .resnet import BasicBlock, Bottleneck, conv1x1, model_urls


class ResNetBC(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetBC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.block_wise_pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def split_block(self, x, div_parts):
        if div_parts == 0:
            return x
        n, c, w, h = x.size()
        block_size = w // div_parts
        l = []
        for i in range(div_parts):
            for j in range(div_parts):
                l.append(x[:, :, i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size])
        x = torch.cat(l, 0)
        return x
    
    def concat_block(self, x, div_parts):
        if div_parts == 0:
            return x
        n, c, w, h = x.size()
        n = n // div_parts ** 2
        r = []
        for i in range(div_parts):
            c = []
            for j in range(div_parts):
                c.append(x[(i * div_parts + j) * n: (i * div_parts + (j + 1)) * n])
            c = torch.cat(c, -1)
            r.append(c)
        x = torch.cat(r, -2)
        return x
    
    def get_dis_block(self, x):
        x = self.block_wise_pool(x)
        f = self.maxpool2(x) * 1

        f_min = torch.min(f, 1)[0].unsqueeze(1)
        f_max = torch.max(f, 1)[0].unsqueeze(1)
        f = (f - f_min) / (f_max - f_min + 1e-9)

        return f

    def forward(self, x, block=[0, 0, 0, 0]):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool1(x)
        if self.training: 
            x = self.split_block(x1, block[0])
            x = self.layer1(x)
            x2 = self.concat_block(x, block[0])

            x = self.split_block(x2, block[1])
            x = self.layer2(x)
            x3 = self.concat_block(x, block[1])

            x = self.split_block(x3, block[2])
            x = self.layer3(x)
            x4 = self.concat_block(x, block[2])

            x = self.split_block(x4, block[3])
            x = self.layer4(x)
            x5 = self.concat_block(x, block[3])
        else:
            x2 = self.layer1(x1)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)
        
        f1 = self.get_dis_block(x3)
        f2 = self.get_dis_block(x4)
        f3 = self.get_dis_block(x5)

        return tuple([x1, x2, x3, x4, x5, f1, f2, f3])


def _resnet_bc(arch, inplanes, planes, cfg, progress, **kwargs):
    
    pretrained = False if "pretrained" not in cfg.keys() else cfg['pretrained']
    del_keys = [] if "del_keys" not in cfg.keys() else cfg['del_keys']
    model = ResNetBC(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        for key in del_keys:
            del state_dict[key]        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return model


def resnet50_bc(cfg, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        cfg (dict): The args of model config node.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_bc('resnet50', Bottleneck, [3, 4, 6, 3], cfg, progress,
                   **kwargs)


def resnet101_bc(cfg, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        cfg (dict): The args of model config node.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_bc('resnet101', Bottleneck, [3, 4, 23, 3], cfg, progress,
                   **kwargs)

