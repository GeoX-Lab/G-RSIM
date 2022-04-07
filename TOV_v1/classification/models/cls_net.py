# -*- coding:utf-8 -*-
'''
Package the torchvision model.

Version 1.0  2020 by QiJi
'''

import torch
import collections
from torch import nn


class ClsNet(nn.Module):
    """ Image-level classification network contains `features` and `classifier`.
    """
    def __init__(self, backbone, num_classes=2, out_dim=None,
                 enhance_features=[], attention_module=None,
                 **kwargs):
        super().__init__()
        if hasattr(backbone, 'model_name'):
            bb_name = backbone.model_name
        else:
            bb_name = backbone._get_name().lower()
        self.model_name = 'ClsNet_' + bb_name

        if 'alexnet' in bb_name:
            self.features = backbone.features
            # self.out_dim = 256
        elif 'resnet' in bb_name or 'resnext' in bb_name:
            bb = nn.Sequential(collections.OrderedDict(list(backbone.named_children())))
            self.features = bb[:8]
            # self.out_dim = 2048
            # if ('resnet18' in bb_name) or ('resnet34' in bb_name):
            #     self.out_dim = 512
        elif 'vgg' in bb_name:
            self.features = backbone.features
            # self.out_dim = 512
        elif 'googlenet' in bb_name:
            backbone = nn.Sequential(collections.OrderedDict(dict(backbone.named_children())))
            self.features = backbone[:16]
            # self.out_dim = 1024
        elif 'inception' in bb_name:
            pass

        if out_dim is not None:
            self.features = nn.Sequential(
                self.features,
                nn.Conv2d(self.out_dim, out_dim, 1),
                nn.BatchNorm2d(out_dim), nn.ReLU(True))
            self.out_dim = out_dim
        else:
            # loop layers and get last conv channels
            for name, m in self.features.named_modules():
                if isinstance(m, torch.nn.Conv2d):
                    self.out_dim = m.out_channels

        self.am = None
        if attention_module is not None:
            raise NotImplementedError('attention is not support this version')

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        # 1. Get feature - [N,C,?,?]
        x = self.features(x)
        if self.am is not None:
            x = self.am(x)

        # 2. Globel avg pooling to get feature of per image
        feature = self.gap(x)  # [N,C,1,1]

        # 3. Get logits
        x = torch.flatten(feature, 1)  # [N,C,1,1] -> [N,C]
        logits = self.classifier(x)

        return logits

