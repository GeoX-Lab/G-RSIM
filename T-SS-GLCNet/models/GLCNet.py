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

import numpy as np
import random
#torch.nn.AvgPool2d(4)



import datetime



def get_patch_q_k(output1, output2, index, patch_size=(16, 16), patch_num=1, outq=None, outk=None):

    q = torch.zeros((output1.shape[0] * patch_num, output2.shape[1]), dtype=output1.dtype, device=output1.device)
    k = torch.zeros((output1.shape[0] * patch_num, output2.shape[1]), dtype=output1.dtype, device=output1.device)
    t = 0
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            if index[i][j][0] != 0:
                temp1 = output1[i, :, index[i][j][0] - patch_size[0] // 2:index[i][j][0] + patch_size[0] // 2,
                        index[i][j][1] - patch_size[1] // 2:index[i][j][1] + patch_size[1] // 2].clone()
                temp2 = output2[i, :, index[i][j][2] - patch_size[0] // 2:index[i][j][2] + patch_size[0] // 2,
                        index[i][j][3] - patch_size[1] // 2:index[i][j][3] + patch_size[1] // 2].clone()
                temp1 = temp1.view(temp1.shape[0], -1)
                temp2 = temp2.view(temp2.shape[0], -1)
                temp11 = torch.mean(temp1, dim=1)
                # temp12=torch.var(temp1,dim=1)
                # temp1=torch.cat([temp11,temp12],dim=0)
                temp21 = torch.mean(temp2, dim=1)
                # temp22=torch.var(temp2,dim=1)
                # temp2=torch.cat([temp21,temp22],dim=0)
                # if torch.sum(torch.isnan(temp11)) > 0 or torch.sum(torch.isnan(temp21)) > 0:
                # print(torch.sum(torch.isnan(temp11)))
                # print(torch.sum(torch.isnan(temp21)))
                q[t, :] = temp11
                k[t, :] = temp21
                t = t + 1
    #print(q)
    if outq is None:

        return q[0:t, :], k[0:t, :]
    else:

        outq.put(q[0:t, :])
        outk.put(k[0:t, :])

def get_patch_q_k1(data_queue,patch_size=(16, 16), patch_num=1, outq=None, outk=None):
    while not data_queue.empty():
        data=data_queue.get()#pop()
        data_queue.task_done()
        output1=data[0].clone()
        output2=data[1].clone()
        index=data[2].clone()
        q = torch.zeros((patch_num, output1.shape[0]), dtype=output1.dtype, device=output1.device)
        k = q.clone()#torch.zeros((patch_num, output2.shape[0]), dtype=output1.dtype, device=output1.device)
        t = 0
       # for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            if index[j][0] != 0:
                temp1 = output1[:, index[j][0] - patch_size[0] // 2:index[j][0] + patch_size[0] // 2,
                        index[j][1] - patch_size[1] // 2:index[j][1] + patch_size[1] // 2].clone()
                temp2 = output2[ :, index[j][2] - patch_size[0] // 2:index[j][2] + patch_size[0] // 2,
                        index[j][3] - patch_size[1] // 2:index[j][3] + patch_size[1] // 2].clone()
                temp1 = temp1.view(temp1.shape[0], -1)
                temp2 = temp2.view(temp2.shape[0], -1)
                temp11 = torch.mean(temp1, dim=1)
                # temp12=torch.var(temp1,dim=1)
                # temp1=torch.cat([temp11,temp12],dim=0)
                temp21 = torch.mean(temp2, dim=1)
                # temp22=torch.var(temp2,dim=1)
                # temp2=torch.cat([temp21,temp22],dim=0)
                # if torch.sum(torch.isnan(temp11)) > 0 or torch.sum(torch.isnan(temp21)) > 0:
                # print(torch.sum(torch.isnan(temp11)))
                # print(torch.sum(torch.isnan(temp21)))
                q[t, :] = temp11
                k[t, :] = temp21
                t = t + 1
        #print(q)
        if outq is None:

            return q[0:t, :], k[0:t, :]
        else:

            outq.put(q[0:t, :])
            outk.put(k[0:t, :])


import threading
import time
from queue import Queue

def get_mask(index,h,w,c,j,patch_size):
    with torch.no_grad():
        out1=torch.zeros((index.shape[0],h,w),device=index.device)
        i1=0
        out2=out1.clone()
        t=torch.ones(index.shape[0],device=index.device)
        for i in range(index.shape[0]):
            if index[i][j][0]==0:
                t[i1]=i
                i1=i1+1
            else:
                out1[i][index[i][j][0]- patch_size[0] // 2: index[i][j][0] + patch_size[0] // 2][index[i][j][1]- patch_size[1] // 2: index[i][j][1] + patch_size[1] // 2]=1
                out2[i][index[i][j][2] - patch_size[0] // 2: index[i][j][2] + patch_size[0] // 2][
                index[i][j][3] - patch_size[1] // 2: index[i][j][3] + patch_size[1] // 2] = 1

        out1=torch.repeat_interleave(out1.unsqueeze(dim=1), repeats=c, dim=1)
        out2 = torch.repeat_interleave(out2.unsqueeze(dim=1), repeats=c, dim=1)
        return out1,out2,t[0:i1].long()


def get_patch_q_k_multi_process1(output1, output2, index, patch_size=(16, 16), patch_num=1, process_num=4):

    if process_num == None:
        q1, k1 = get_patch_q_k(output1, output2, index, patch_size=patch_size, patch_num=patch_num)
    else:
        data_queue=Queue()
        for i in range(output1.shape[0]):
            data_queue.put([output1[i,:,:,:],output2[i,:,:,:],index[i,:,:]])
        q = Queue()
        k = Queue()
        #n = int(output1.shape[0] / process_num)
        # print(n)
        threads = []
        # tb = threading.Thread(target=get_patch_q_k, args=(
        # output1.clone(), output2.clone(), target1.clone(), target2.clone(), patch_size, patch_num, q, k))

        for i in range(process_num):
            t = threading.Thread(target=get_patch_q_k1, args=(
                data_queue, patch_size, patch_num, q,k))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
        data_queue.join()
        q1 = []
        k1 = []

        flag = True
        while not q.empty():
            if flag:
                q1 = q.get()
                k1 = k.get()
                flag = False
            else:
                q1 = torch.cat([q1, q.get()], dim=0)
                k1 = torch.cat([k1, k.get()], dim=0)

    return q1, k1

class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        #self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        #self.dropout3 = nn.Dropout(0.1)
        self.conv41 = nn.Conv2d(256, class_num, kernel_size=1)

        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4 = F.interpolate(
            x,
            size=low_level_feature.size()[2:4],
            mode='bilinear',
            align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        #x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        #x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv41(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self,
                 num_classes=2,
                 in_channels=3,
                 arch='resnet101',
                 output_stride=16,
                 bn_momentum=0.9,
                 freeze_bn=False,noStyle=False,noGlobal=False,noLocal=False,
                 pretrained=False,patch_size=16,patch_num=4,patch_out_channel=False,pross_num=28,
                 **kwargs):
        super(DeepLab, self).__init__(**kwargs)
        self.model_name = 'deeplabv3plus_' + arch

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
        self.decoder = Decoder(num_classes, bn_momentum)
        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
        self.patch_num=patch_num
        self.patch_size=patch_size
        self.pross_num=pross_num
        self.noStyle=noStyle
        self.noGlobal=noGlobal
        self.noLocal=noLocal
        # projection head
        '''
        self.proj = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 10, 1, bias=True)
        )
        '''
        if self.noStyle:
            self.proj = nn.Sequential(nn.Linear(256 , 256), nn.ReLU(), nn.Linear(256, num_classes))
        else:
            self.proj =nn.Sequential(nn.Linear(256*2, 256), nn.ReLU(), nn.Linear(256,num_classes))
        if not patch_out_channel:
            patch_out_channel=num_classes
        self.proj1 = nn.Sequential(nn.Linear(num_classes, num_classes), nn.ReLU(), nn.Linear(num_classes, patch_out_channel))
    def forward(self, input,input1,index):
        x,low_level_features = self.backbone(input)
        #print(low_level_features.size()),56
        x = self.encoder(x)
        if not self.noLocal:
            predict = self.decoder(x.clone(), low_level_features)
            predict = F.interpolate(
                predict,
                size=input.size()[2:4],
                mode='bilinear',
                align_corners=True)
        if not self.noGlobal:
            if self.noStyle:
                x=self.avgpool(x)
                x = torch.flatten(x, 1)
            else:

                x1=self.avgpool(x)
                #print(x.size())
                x1 = torch.flatten(x1, 1)
                x=x.view(x.shape[0],x.shape[1],-1)
                x=torch.var(x,dim=2)
                x=torch.cat([x1,x],dim=1)

            q=self.proj(x)
        x, low_level_features = self.backbone(input1)

        x = self.encoder(x)
        if not self.noLocal:
            predict1 = self.decoder(x.clone(), low_level_features)
            predict1 = F.interpolate(
                predict1,
                size=input.size()[2:4],
                mode='bilinear',
                align_corners=True)
        if not self.noGlobal:
            if self.noStyle:
                x=self.avgpool(x)
                x = torch.flatten(x, 1)
            else:
                x1=self.avgpool(x)
                x1 = torch.flatten(x1, 1)
                x=x.view(x.shape[0],x.shape[1],-1)
                x=torch.var(x,dim=2)
                x=torch.cat([x1,x],dim=1)
            #print(x)
            k=self.proj(x)
        #del x,x1, low_level_features
        '''
        q1,k1=None,None
        for i in range(index.shape[1]):
            t1,t2,t=get_mask(index,input.shape[2],input.shape[3],predict.shape[1],i,(self.patch_size,self.patch_size))

            t1=predict*t1.float()
            t2=predict1*t2.float()
            t1=self.avgpool(t1)*(input.shape[2]*input.shape[2])/(self.patch_size*self.patch_size)
            t2 = self.avgpool(t2) * (input.shape[2] * input.shape[2]) / (self.patch_size * self.patch_size)
            t1=torch.index_select(t1, dim=0, index=t)
            t2=torch.index_select(t2, dim=0, index=t)
            if i==0:
                q1=t1
                k1=t2
            else:
                q1 = torch.cat([q1, t1], dim=0)
                k1 = torch.cat([k1, t2], dim=0)
        '''

        #predict = self.decoder(x, low_level_features)
        if not self.noLocal:
            q1, k1 = get_patch_q_k_multi_process1(predict, predict1, index ,patch_size=(self.patch_size, self.patch_size), patch_num=self.patch_num,
                                                 process_num=self.pross_num)

            q1=self.proj1(q1)
            k1=self.proj1(k1)
        if self.noGlobal:
            return q1,k1
        elif self.noLocal:
            return q,k
        else:
            return q,k,q1,k1#predict,predict1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()





def build_model(num_classes=5, in_channels=3,pretrained=False,arch='resnet101',patch_num=4,patch_size=16,patch_out_channel=False,pross_num=28,noStyle=False,noGlobal=False,noLocal=False):
    model = DeepLab(num_classes=num_classes, in_channels=in_channels,pretrained=pretrained,arch=arch,patch_num=patch_num,patch_size=patch_size,patch_out_channel=patch_out_channel,pross_num=pross_num,noStyle=noStyle,noGlobal=noGlobal,noLocal=noLocal)
    return model

if __name__ == "__main__":
    model = DeepLab(
        output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
    print(m)
