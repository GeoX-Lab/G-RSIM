# -*- coding:utf-8 -*-
'''
Package the torchvision model into FCN.
'''

import torch
import collections
from torch import nn
from torch.nn import functional as F

from torchvision.models._utils import IntermediateLayerGetter


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class FCNHead_simple(nn.Sequential):
    def __init__(self, in_channels, channels):
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        ]

        super(FCNHead_simple, self).__init__(*layers)


class UpBlock(nn.Sequential):
    def __init__(self, in_channels, channels):
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            # nn.ReLU(),
        ]
        super(UpBlock, self).__init__(*layers)

    def forward(self, x):
        for mod in self:
            x = mod(x)
        return F.interpolate(
            x, scale_factor=2.0, mode='bilinear', align_corners=False)


class FCN(nn.Module):
    """ Fully conv network contains `features` and `classifier`.
    """
    def __init__(self, backbone, num_classes=2, aux=None,
                 **kwargs):
        super().__init__()
        if hasattr(backbone, 'model_name'):
            bb_name = backbone.model_name
        else:
            bb_name = backbone._get_name().lower()
        self.model_name = 'FCN_' + bb_name

        if 'alexnet' in bb_name:
            self.features = backbone.features
            # self.out_dim = 256
        elif 'resnet' in bb_name or 'resnext' in bb_name:
            return_layers = {'layer4': 'out'}
            if aux:
                self.aux_dim = 1024
                if ('resnet18' in bb_name) or ('resnet34' in bb_name):
                    self.out_dim = 256
                return_layers['layer3'] = 'aux'
            self.features = IntermediateLayerGetter(
                backbone, return_layers=return_layers)
        elif 'vgg' in bb_name:
            self.features = backbone.features
            # self.out_dim = 512
        elif 'googlenet' in bb_name:
            backbone = nn.Sequential(collections.OrderedDict(dict(backbone.named_children())))
            self.features = backbone[:16]
            # self.out_dim = 1024
        elif 'inception' in bb_name:
            pass

        self.out_dim = None
        if hasattr(backbone, 'out_dim'):
            self.out_dim = backbone.out_dim
        else:
            # loop layers and get last conv channels
            for name, m in self.features.named_modules():
                if isinstance(m, torch.nn.Conv2d):
                    self.out_dim = m.out_channels

        self.aux_classifier = None
        if aux:
            self.aux_classifier = FCNHead(self.aux_dim, num_classes)

        self.classifier = FCNHead(self.out_dim, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.features(x)

        result = collections.OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class FCN_d8(FCN):
    """ Fully conv network contains `features` and `classifier`.
    """
    def __init__(self, backbone, num_classes=2, aux=None,
                 **kwargs):
        super(FCN_d8, self).__init__(backbone, num_classes, aux, **kwargs)

        self.model_name = self.model_name.replace('FCN', 'FCN_d8')

        if aux:
            self.aux_classifier = FCNHead(self.aux_dim, num_classes)

        self.uplayer1 = UpBlock(self.out_dim, self.out_dim//2)  # d4
        self.uplayer2 = UpBlock(self.out_dim//2, self.out_dim//4)  # d2

        self.classifier = FCNHead_simple(self.out_dim//4, num_classes)  # d1

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.features(x)

        result = collections.OrderedDict()
        x = features["out"]
        x = self.uplayer1(x)  # d4
        x = self.uplayer2(x)  # d2
        x = self.classifier(x)  # d1
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result
