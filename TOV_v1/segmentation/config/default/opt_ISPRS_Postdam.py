#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about ISPRS_Postdam dataset, segmentation mode
'''
from config.category import ISPRS


class Config(object):
    ''' Fixed initialization Settings '''
    # Path and file
    root = ''
    data_dir = root + 'segmentation/Seg_data/ISPRS_Postdam'
    ckpt = root + 'Task_models/ISPRS_Postdam'
    env = 'ISPRS_Postdam'  # visdom 环境

    train_val_ratio = None
    train_scale = 1  # Scale of training set reduction
    val_scale = 0.1
    # Model related arguments
    sepoch = 1  # use to continue from a checkpoint
    pretrained = False  # use pre-train checkpoint

    # Optimiztion related arguments
    batch_size = 16  # batch size
    max_epochs = None  # 16-[256,256] dataset only need 8~9 epoch
    ckpt_freq = 0
    learning_rate = 4e-4  # initial learning rate
    lr_decay_rate = 0.5
    lr_decay_steps = 20
    lr_scheduler = 'cosine'  # pre epoch
    warmup = 0  # if warmup > 0, use warmup strategy and end at warmup
    weight_decay = 1e-5  # L2 loss
    optimizer = ['adam', 'sgd', 'lars'][1]
    loss = ['CrossEntropyLoss'][0]
    loss_weight = None

    # Data related arguments
    num_workers = 4  # number of data loading workers
    dtype = ['RGB'][0]
    bl_dtype = [''][0]
    in_channel = 0

    input_size = (256, 256)  # final input size of network(random-crop use this)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    category = ISPRS()
    classes = category.names
    class_dict = category.table
    num_classes = len(classes)

    reduce_zero_label = False
    ignore_index = 0
