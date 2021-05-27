# -*- coding:utf-8 -*-
'''
Abstract.

Version 1.0  2020-07-06 14:32:04
by QiJi Refence:
TODO:
'''

import os
import time
from collections import deque
# from itertools import chain

import numpy as np
import torch
from PIL import Image
# import torch.nn as nn
import torch.nn.parallel
# import torch.distributed as dist
import torch.optim
import torch.utils.data
try:
    from apex import amp
except ImportError:
    amp = None






def Jigsaw_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use inpainting pretext task. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = data['image'], data['label']


        input = input.cuda(non_blocking=True)  # [NCHW]
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
        '''
        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                predict = output.argmax(1)
                acc = predict.eq(target).float().mean().cpu().numpy()
                dltrain.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} acc: {:.3f} Time: {:.2f}s\r'.format(
                        epoch, args.mepoch, i, loss.item(), acc, time.time() - tic), '\t')
                tic = time.time()  # update time
        '''

    return np.mean(loss_hist)
import datetime
def GLCNet_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use GLCNet. '''


    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    #tic = time.time()
    for i, data in enumerate(train_loader):

        optimizer.zero_grad()

        input1, input2,index=data['image'], data['image1'],data['index']
        input1 = input1.cuda(non_blocking=args.non_blocking)  # [NCHW]
        input2 = input2.cuda(non_blocking=args.non_blocking)  # [NCHW]
        index = index.cuda(non_blocking=args.non_blocking)  # [NCHW]
        if args.self_mode==12:
            q1, k1 = model(input1, input2, index)
            loss = criterion[1](q1, k1)
        elif args.self_mode==13:
            q, k = model(input1, input2, index)
            loss = criterion[0](q, k)
        else:

            q, k ,q1,k1= model(input1, input2,index)  #slow# ,output,output1

            loss = 0.5*criterion[0](q, k)+0.5*criterion[1](q1,k1)

        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()  #slow
        #loss.backward()
        optimizer.step()


        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    return np.mean(loss_hist)

def SimCLR_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use SimCLR. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    #tic = time.time()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        input1, input2=data['image'], data['image1']
        if args.gpu is not None:
            input1 = input1.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            input2 = input2.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
        else:
            input1 = input1.cuda(non_blocking=args.non_blocking)  # [NCHW]
            input2 = input2.cuda(non_blocking=args.non_blocking)  # [NCHW]
        # print('0')
        # print(datetime.datetime.now())
        q, k = model(input1, input2)  # ,output,output1
        # print('1')
        # print(datetime.datetime.now())
        loss = criterion(q, k)
        # print('2')
        # print(datetime.datetime.now())
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #print('3')
       # print(datetime.datetime.now())
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if not args.use_mp or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                #dltrain.train_log(
                  #  ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} ({:.3f}) Time: {:.2f}s\r'.format(
                     #   epoch, args.mepoch, i, loss.item(), np.mean(loss_hist), time.time() - tic), '\t')
                tic = time.time()  # update time
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    #if args.vis and (epoch == args.sepoch or epoch == args.mepoch) and (
            #not args.mp_distributed or args.gpu == 0):
        #log_data_for_ss(input1, input2, epoch, args)
    return np.mean(loss_hist)
import torch.nn.functional as F
def Test_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use SimCLR. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        input1, input2=data['image'], data['image1']
        label1, label2 = data['label'], data['label1']
        if args.gpu is not None:
            input1 = input1.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            input2 = input2.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            label1 = label1.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            label2 = label2.cuda(args.gpu, non_blocking=args.non_blocking)
        else:
            input1 = input1.cuda(non_blocking=args.non_blocking)  # [NCHW]
            input2 = input2.cuda(non_blocking=args.non_blocking)  #
            label1 = label1.cuda(non_blocking=args.non_blocking)  # [NCHW]
            label2 = label2.cuda(non_blocking=args.non_blocking)


        # [NCHW]
        q, k,output1,output2 = model(input1, input2)  # ,output,output1
        '''
        label1 = F.interpolate(
            label1.unsqueeze(1),size=output1.size()[2:4],mode='nearest').squeeze(1)
        label2 = F.interpolate(
            label2.unsqueeze(1), size=output2.size()[2:4], mode='nearest').squeeze(1)
        '''
        loss = criterion[0](q, k)+criterion[1](output1,label1)+criterion[1](output2,label2)
        del q,k,output1,output2
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    #if args.vis and (epoch == args.sepoch or epoch == args.mepoch) and (
            #not args.mp_distributed or args.gpu == 0):
        #log_data_for_ss(input1, input2, epoch, args)
    return np.mean(loss_hist)

def MoCov2_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use SimCLR. '''
    model.train()
    m=0.999
    k=2048
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        input1, input2=data['image'], data['image1']
        #if args.gpu is not None:
            #input1 = input1.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
            #input2 = input2.cuda(args.gpu, non_blocking=args.non_blocking)  # [NCHW]
        #else:
        input1 = input1.cuda(non_blocking=args.non_blocking)  # [NCHW]
        input2 = input2.cuda(non_blocking=args.non_blocking)  # [NCHW]
        logits, labels = model(input1, input2)  # ,output,output1
        loss = criterion(logits, labels)
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    #if args.vis and (epoch == args.sepoch or epoch == args.mepoch) and (
            #not args.mp_distributed or args.gpu == 0):
        #log_data_for_ss(input1, input2, epoch, args)
    return np.mean(loss_hist)
def get_erase_mask(bs, opts, erase_shape=[16, 16], erase_count=16):
    #Random block
    H, W = opts.input_size
    masks = torch.ones((bs, opts.n_channels, H, W))
    for n in range(bs):
        for _ in range(erase_count):
            row = np.random.randint(0, H - erase_shape[0] - 1)
            col = np.random.randint(0, W - erase_shape[1] - 1)
            masks[n, :, row: row+erase_shape[0], col: col+erase_shape[1]] = 0
    return masks
def get_central_mask(bs, opts, erase_ratio=1/2):
    #Central region
    H, W = opts.input_size
    masks = torch.ones((bs, opts.n_channels, H, W))
    eH, eW = int(H*erase_ratio), int(W*erase_ratio)
    row_st = (H - eH) // 2
    col_st = (W - eW) // 2
    masks[:, :, row_st: row_st+eH, col_st: col_st+eW] = 0
    return masks

def Inpaiting_train(train_loader, model, criterion, optimizer, epoch, args,summary_writer=None):
    ''' One epoch training use SimCLR. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    lr=optimizer.param_groups[0]['lr']
    masks = get_erase_mask(args.self_batch_size, args)
    if args.use_gpu:
        masks = masks.cuda(non_blocking=args.non_blocking)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        input=data['image']

        input = input.cuda(non_blocking=args.non_blocking)  # [NCHW]
        #input2 = input2.cuda(non_blocking=args.non_blocking)  # [NCHW]
        output = model(input * masks)  # ,output,output1
        mse_loss = criterion(output, input)
        loss_rec = torch.sum(mse_loss * (1 - masks)) / torch.sum(1 - masks)
        loss_con = torch.sum(mse_loss * masks) / torch.sum(masks)
        # loss = torch.sum(mse_loss*(1-masks)) / torch.sum(1-masks)
        loss = 0.99 * loss_rec + 0.01 * loss_con
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if summary_writer is not None:
            if i % args.print_freq == 0:
                step = (epoch - 1) * len(train_loader) + i
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', np.mean(loss_hist), step)
    #if args.vis and (epoch == args.sepoch or epoch == args.mepoch) and (
            #not args.mp_distributed or args.gpu == 0):
        #log_data_for_ss(input1, input2, epoch, args)
    return np.mean(loss_hist)





if __name__ == '__main__':
    # main()
    pass

