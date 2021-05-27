# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

# import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab_utils import  ResNet
from models.deeplab_utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplab_utils.encoder import Encoder




class DeepLab(nn.Module):
    def __init__(self,
                 num_classes=2,
                 in_channels=3,
                 arch='resnet101',
                 output_stride=16,
                 bn_momentum=0.9,
                 freeze_bn=False,
                 pretrained=False,puzzle=3,
                 **kwargs):
        super(DeepLab, self).__init__(**kwargs)
        self.model_name = 'deeplabv3plus_' + arch

        #num_classes=puzzle**2

        # Setup arch
        if arch == 'resnet18':
            NotImplementedError('resnet18 backbone is not implemented yet.')
        elif arch == 'resnet34':
            NotImplementedError('resnet34 backbone is not implemented yet.')
        elif arch == 'resnet50':
            self.backbone = ResNet.resnet50(bn_momentum, pretrained)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif arch == 'resnet101':
            self.backbone = ResNet.resnet101(bn_momentum, pretrained)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = Encoder(bn_momentum, output_stride)
        #self.decoder = Decoder(num_classes, bn_momentum)
        self.avgpool =  nn.AdaptiveAvgPool2d((3, 3))
        self.fc5 = nn.Sequential(
            nn.Linear(256 * puzzle * puzzle, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * (puzzle**2), 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
        )
        # projection head
        '''
        self.proj = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 10, 1, bias=True)
        )
        '''
        self.proj =nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256,num_classes))
    def forward(self, x):
        N, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(T):
            z,_ = self.backbone(x[i])  # 2x2
            z = self.encoder(z)
            z=self.avgpool(z)
            z = self.fc5(z.view(N, -1))
            z = z.view([N, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc6(x.view(N, -1))
        logist = self.classifier(x)
        return logist
       

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()




def build_model(num_classes=5, in_channels=3,pretrained=False,arch='resnet101'):
    model = DeepLab(num_classes=num_classes, in_channels=in_channels,pretrained=pretrained,arch=arch)
    return model

if __name__ == "__main__":
    model = DeepLab(
        output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
