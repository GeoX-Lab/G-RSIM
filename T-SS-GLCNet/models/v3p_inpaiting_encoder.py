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


class Context_Encoder0(nn.Module):
    def __init__(self, net, in_size=224):
        super().__init__()
        self.model_name = 'Context_encoder0_' + net.model_name
        # self.insize = in_size

        # Encoder
        self.feature = net.feature  # .copy()

        # replace Relu to LeakyRelu
        for name, module in self.feature.named_children():
            if isinstance(module, nn.ReLU):
                module = nn.LeakyReLU(0.2)

        self.out_dim = net.out_dim

        # Channel-wise fully-connected layer
        # if 'resnet' in net.model_name:
        #     md_size = 7
        # elif 'alexnet' in net.model_name:
        #     md_size = 6
        # self.channel_wise_fc = nn.Parameter(
        #     torch.rand(self.out_dim, md_size*md_size, md_size*md_size)
        # )
        # nn.init.normal_(self.channel_wise_fc, 0., 0.005)
        # self.dropout_cwfc = nn.Dropout(0.5, inplace=True)
        # self.conv_cwfc = nn.Conv2d(self.out_dim, self.out_dim, 1)

        # Decoder
        # if self.out_dim == 2048:
        #     next_dim = 512
        # else:
        #     next_dim = 256

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.out_dim, 128, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # size: 128 x 11 x 11
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # size: 64 x 21 x 21
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # size: 64 x 41 x 41
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # size: 32 x 81 x 81
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            # size: 3 x 161 x 161
            # reszie to 227, 227
            nn.Tanh()
        )

    def forward(self, x):
        insize = x.size()[2:4]
        x = self.feature(x)  # s/32

        # N, C, H, W = x.size()[:]
        # x = x.view(N, C, -1)  # [N,C,H,W] -> [N,C,HW]
        # x = x.permute(1, 0, 2)  # [N,C,HW] -> [C,N,HW]
        # x = torch.bmm(x, self.channel_wise_fc)
        # x = x.permute(1, 0, 2)
        # x = self.dropout_cwfc(x)
        # x = x.view(N, C, H, W)
        # x = self.conv_cwfc(x)

        x = self.decoder1(x)  # s/16
        x = self.decoder2(x)  # s/8
        x = self.decoder3(x)  # s/4
        x = self.decoder4(x)  # s/2
        x = self.decoder5(x)  # s/1
        # x = nn.functional.interpolate(x, size=self.insize, mode='nearest')
        x = nn.functional.interpolate(x, size=insize, mode='nearest')
        return x


class Context_Encoder(nn.Module):
    def __init__(self, net, in_channels=3):
        super().__init__()
        self.model_name = 'Context_encoder2_' + net.model_name
        # self.insize = in_size
        self.out_dim = net.out_dim
        # Encoder
        self.conv1 = nn.Sequential(
            net.feature.conv1,
            net.feature.bn1,
            net.feature.relu,
        )  # 64, s/2
        self.layer1 = net.feature.layer1  # 256, s/4
        self.layer2 = net.feature.layer2  # 512, s/8
        self.layer3 = net.feature.layer3  # 1024, s/8
        self.layer4 = net.feature.layer4  # 2048, s/16

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(2048, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2,
                               padding=2, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256+1024, 256, 5, 2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256+512, 256, 3, 2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, 2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=7, padding=3, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        s_1 = x.size()[2:4]
        x = self.conv1(x)  # s/2
        # s_2 = x.size()[2:4]
        # x = self.maxpool(x)
        e1 = self.layer1(x)  # 256, s/2
        e2 = self.layer2(e1)  # 512, s/4
        s_4 = x.size()[2:4]
        e3 = self.layer3(e2)  # 1048, s/8
        s_8 = e3.size()[2:4]
        e4 = self.layer4(e3)  # 2048, s/16
        e4 = self.bottleneck_conv(e4)  # 256

        # Decode
        d4 = self.decoder4(e4)  # 256, s/8
        if d4.shape[2:4] != e3.shape[2:4]:
            d4 = F.interpolate(d4, size=s_8, mode='nearest')
        d4_cat = torch.cat((d4, e3), dim=1)  # 256+1024

        d3 = self.decoder3(d4_cat)  # 256, s/4
        if d3.shape[2:4] != e2.shape[2:4]:
            d3 = F.interpolate(d3, size=s_4, mode='nearest')
        d3_cat = torch.cat((d3, e2), dim=1)  # 256+512

        d2 = self.decoder2(d3_cat)  # 256, s/2
        d1 = self.decoder1(d2)  # 64, s/1
        if d1.shape[2:4] != s_1:
            d1 = F.interpolate(d1, size=s_1, mode='nearest')
        x = self.conv_final(d1)
        # x = nn.functional.interpolate(x, size=self.insize, mode='nearest')
        return x



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
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2,
                               padding=2, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 1024, 256, 5, 2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 512, 256, 3, 2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, 2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=7, padding=3, bias=True),
            nn.Tanh()
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
        #self.proj =nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256,num_classes))
    def forward(self, x):




        # Encode
        s_1 = x.size()[2:4]

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)# s/2

        e1 = self.backbone.layer1(x)  # 256, s/2
        e2 = self.backbone.layer2(e1)  # 512, s/4
        s_4 = x.size()[2:4]
        e3 = self.backbone.layer3(e2)  # 1048, s/8
        s_8 = e3.size()[2:4]
        e4 = self.backbone.layer4(e3)  # 2048, s/16
        e4=self.encoder(e4)
        #e4 = self.bottleneck_conv(e4)  # 256

        # Decode
        d4 = self.decoder4(e4)  # 256, s/8
        if d4.shape[2:4] != e3.shape[2:4]:
            d4 = F.interpolate(d4, size=s_8, mode='nearest')
        d4_cat = torch.cat((d4, e3), dim=1)  # 256+1024

        d3 = self.decoder3(d4_cat)  # 256, s/4
        if d3.shape[2:4] != e2.shape[2:4]:
            d3 = F.interpolate(d3, size=s_4, mode='nearest')
        d3_cat = torch.cat((d3, e2), dim=1)  # 256+512

        d2 = self.decoder2(d3_cat)  # 256, s/2
        d1 = self.decoder1(d2)  # 64, s/1
        if d1.shape[2:4] != s_1:
            d1 = F.interpolate(d1, size=s_1, mode='nearest')
        x = self.conv_final(d1)
        # x = nn.functional.interpolate(x, size=self.insize, mode='nearest')
        return x


       

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
