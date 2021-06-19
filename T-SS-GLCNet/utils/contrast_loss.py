# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torch.autograd as autograd


def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    #mask = mask.type(torch.uint8)
    return mask


def get_pos_and_neg_mask(bs):
    ''' Org_NTXentLoss_mask '''
    zeros = torch.zeros((bs, bs), dtype=torch.uint8)
    eye = torch.eye(bs, dtype=torch.uint8)
    pos_mask = torch.cat([
        torch.cat([zeros, eye], dim=0), torch.cat([eye, zeros], dim=0),
    ], dim=1)
    neg_mask = _get_correlated_mask(bs)
    #(torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(2*bs, dtype=torch.uint8))
    pos_mask = mask_type_transfer(pos_mask)
    neg_mask = mask_type_transfer(neg_mask)
    return pos_mask, neg_mask


class NTXentLoss(nn.Module):
    """ NTXentLoss

    Args:
        tau: The temperature parameter.
    """

    def __init__(self,
                 bs,gpu,
                 tau=1,
                 cos_sim=False,
                 use_gpu=True,
                 eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.name = 'NTXentLoss_Org'
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.gpu = gpu
        self.eps = eps
        self.bs=bs

        if cos_sim:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            self.name += '_CosSim'

        # Get pos and neg mask
        self.pos_mask, self.neg_mask = get_pos_and_neg_mask(bs)

        if use_gpu:
            self.pos_mask = self.pos_mask.cuda(gpu)
            self.neg_mask = self.neg_mask.cuda(gpu)
        print(self.name)

    def forward(self, zi, zj):
        '''
        input: {'zi': out_feature_1, 'zj': out_feature_2}
        target: one_hot lbl_prob_mat
        '''
        zi, zj = F.normalize(zi, dim=1), F.normalize(zj, dim=1)
        bs = zi.shape[0]

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j
        # [2*bs, 2*bs] -  pairwise similarity
        if self.use_cos_sim:
            sim_mat = torch.exp(self.cosine_similarity(
                z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau)  # s_(i,j)
        else:
            sim_mat = torch.exp(torch.mm(z_all, z_all.t().contiguous()) / self.tau)  # s_(i,j)
        # if bs!=self.bs:
        #     pos_mask, neg_mask = get_pos_and_neg_mask(bs)
        #     pos_mask, neg_mask=pos_mask.cuda(self.gpu), neg_mask(self.gpu)
        #     sim_pos = sim_mat.masked_select(pos_mask).view(2 * bs).clone()
        #     # [2*bs, 2*bs-1]
        #     sim_neg = sim_mat.masked_select(neg_mask).view(2 * bs, -1)
        # else:

        #pos = torch.sum(sim_mat * self.pos_mask, 1)
        #neg = torch.sum(sim_mat * self.neg_mask, 1)
        #loss = -(torch.mean(torch.log(pos / (pos + neg))))
        sim_pos = sim_mat.masked_select(self.pos_mask).view(2 * bs).clone()
        # [2*bs, 2*bs-1]
        sim_neg = sim_mat.masked_select(self.neg_mask).view(2 * bs, -1)
    # Compute loss
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()


        return loss

def _get_correlated_mask(batch_size):
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask)#.byte()#.type(torch)
    return mask#.to(self.device)

def get_contrast_loss(name, **kwargs):
    if name == 'NTXentLoss':
        criterion = NTXentLoss

    return criterion(**kwargs)


def main():

    pass


if __name__ == '__main__':
    main()
