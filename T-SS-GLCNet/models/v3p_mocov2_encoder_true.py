# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

import os
# from torchsummary import summary
#from models.deeplab.ResNet101 import resnet101
#from models.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

#from models.deeplab.encoder import Encoder
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab_utils import  ResNet
from models.deeplab_utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplab_utils.encoder import Encoder




class DeepLab0(nn.Module):
    def __init__(self,
                 num_classes=2,
                 in_channels=3,
                 arch='resnet101',
                 output_stride=16,
                 bn_momentum=0.9,
                 freeze_bn=False,
                 pretrained=False,
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
        # self.decoder = Decoder(num_classes, bn_momentum)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # projection head
        '''
        self.proj = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 10, 1, bias=True)
        )
        '''
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, input, input1):
        x, _ = self.backbone(input)
        # print(low_level_features.size()),56
        x = self.encoder(x)
        # print(x.size()),14
        x = self.avgpool(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        q = self.proj(x)
        x, _ = self.backbone(input1)

        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        k = self.proj(x)
        # predict = self.decoder(x, low_level_features)

        return q, k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
class DeepLab(nn.Module):
    def __init__(self,
         
                 in_channels=3,
                 arch='resnet101',
                 output_stride=16,
                 bn_momentum=0.9,
                 freeze_bn=False,
                 pretrained=False,
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
        # self.decoder = Decoder(num_classes, bn_momentum)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # projection head
        '''
        self.proj = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 10, 1, bias=True)
        )
        '''
        #self.proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, input):
        x, _ = self.backbone(input)
        # print(low_level_features.size()),56
        x = self.encoder(x)
        # print(x.size()),14
        x = self.avgpool(x)

        # predict = self.decoder(x, low_level_features)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, in_channels=3, out_dim=10, m=0.99, T=0.07, pretrained=True, bn_momentum=0.9,
                 num_classes1=30,arch='resnet50',K=65536):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07), dim=128, K=65536, m=0.999, T=0.07, mlp=False
        """
        super(MoCo, self).__init__()
        self.m = m
        self.T = T
        self.K=K
        dim_mlp = 256
        self.out_dim =out_dim
        self.encoder_q = DeepLab(in_channels=in_channels, pretrained=pretrained,arch=arch)
        self.encoder_k = DeepLab(in_channels=in_channels, pretrained=pretrained,arch=arch)
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))
        self.proj1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))
        #self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 64), nn.Linear(64, 64))
        #self.fc1 = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 64), nn.Linear(64, 64))
        # nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, 1),nn.BatchNorm2d(dim_mlp),nn.ReLU(inplace=True),
        # nn.Conv2d(dim_mlp, out_dim, 1),nn.Conv2d(out_dim, out_dim, 1))
        # self.fc1 = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, 1),nn.BatchNorm2d(dim_mlp),nn.ReLU(inplace=True),
        # nn.Conv2d(dim_mlp, out_dim, 1),nn.Conv2d(out_dim, out_dim, 1))
        #self.avgpool = nn.AvgPool2d(28)  # , stride=1
        #self.avgpool1 = nn.AvgPool2d(28)
        # self.outconv=nn.Sequential(nn.Conv2d(256,256,kernel_size=1), nn.ReLU(),nn.Conv2d(256,64,kernel_size=1),nn.Conv2d(64,64,kernel_size=1))
        # self.outconv1=nn.Sequential(nn.Conv2d(256,256,kernel_size=1), nn.ReLU(),nn.Conv2d(256,64,kernel_size=1),nn.Conv2d(64,64,kernel_size=1))

        # self.decoder1 = Decoder(num_classes, bn_momentum)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.proj.parameters(), self.proj1.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.register_buffer("queue", torch.randn(out_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.proj.parameters(), self.proj1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, im_q, im_k):
         # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = torch.flatten(q, 1)
        q = self.proj(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k=torch.flatten(k,1)
            k=self.proj1(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
           # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
       
      
        return logits, labels  # logits, labels
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
def build_model(num_classes=5, in_channels=3,pretrained=False,arch='resnet50',m=0.999,K=1024,T=0.07):
    model = MoCo(out_dim=num_classes, in_channels=in_channels,pretrained=pretrained,arch=arch,m=m,K=K,T=T)
    return model


if __name__ == "__main__":
    model = DeepLab(
        output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    # print(model)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # summary(model, (3, 513, 513))
    # for m in model.named_modules():
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
