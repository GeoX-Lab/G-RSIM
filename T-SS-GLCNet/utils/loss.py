# -*- coding:utf-8 -*-
'''
Some custom loss functions for PyTorch.

Version 1.0  2018-11-02 15:15:44

cross_entropy2d() and multi_scale_cross_entropy2d() are not written by me.
'''
from torch import nn
import os
import cv2
import torch
import numpy as np
# import torch.nn as nn
import torch.nn.functional as F
# import torch.autograd as autograd


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.CrossEntropyLoss(weight=self.weight)
        #self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index

            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
        

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(
        log_p, target, ignore_index=250, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def multi_scale_cross_entropy2d(input,
                                target,
                                weight=None,
                                size_average=True,
                                scale_weight=None):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp),
                                 torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average)

    return loss


def jaccard_loss(input, target):
    ''' Soft IoU loss. Note: 未测试多类（2类以上，不包括2类）情况下是否正确
    Args:
        input - net output tensor, one_hot [NCHW]
        target - gt label tensor, one_hot [NCHW]
    '''
    n, c, h, w = input.size()
    # nt, ht, wt = target.size()
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    input = torch.sigmoid(input)
    # Expand target tensor dim
    target = torch.zeros(n, 2, h, w).scatter_(dim=1, index=target, value=1)
    intersection = input * target  # #[NCHW]  # #相同为input，不同为0
    # input1 = input.cpu().detach().numpy()
    # target1  = target.cpu().detach().numpy()
    union = input + target - intersection  # #相同为1，不同为input
    iou = intersection / union  # #相同为input/1，不同为0
    # iou1 = iou.cpu().detach().numpy()
    return (intersection / union).sum() / (n*h*w)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

def main():

    # 预测值f(x) 构造样本，神经网络输出层
    # input_tensor = torch.ones([3, 2, 5, 5], dtype=torch.float64)
    # tmp_mat = torch.ones([5, 5], dtype=torch.float64)
    # input_tensor[0, 0, :, :] = tmp_mat * 0.5
    # input_tensor[1, 1, :, :] = tmp_mat * 0.5
    # input_tensor[2, 1, :, :] = tmp_mat * 0.5
    # label = torch.argmax(input_tensor, 3)
    # print(label[0])
    # print(label[1])
    # print(label.size())
    # [0.8, 0.2] * [1, 0]: 0.8 / (0.8+0.2 + 1 - 0.8) = 0.8 / 1.2 = 2/3
    # [0.4, 0.6] * [1, 0]: 0.4 / (2 - 0.4) = 0.4 / 1.6 = 1/4
    # [0.0, 1.0] * [0, 1]: 0

    # 真值y
    # labels = torch.LongTensor([0, 1, 4, 7, 3, 2]).unsqueeze(1)
    # print(labels.size())
    # one_hot = torch.zeros(6, 8).scatter_(dim=1, index=labels, value=1)
    # print(one_hot)

    # target_tensor = torch.ones([3, 5, 5], dtype=torch.int64).unsqueeze(1)
    # target_tensor = torch.zeros(3, 2, 5, 5).scatter_(1, target_tensor, 1)
    # print(target_tensor.size())
    # J = input_tensor * target_tensor

    p = np.array([0.8, 0.2])
    t = np.array([1, 0])
    print()
    pass


if __name__ == '__main__':
    main()
