#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about EuroSAT dataset, classification mode
URL: https://github.com/phelber/eurosat
'''
from config.category import EuroSAT


class Config(object):
    ''' Fixed initialization Settings '''
    # Path and file
    root = ''
    data_dir = root + 'classification/Cls_data/EuroSATRGB'
    ckpt = root + 'Task_models/EuroSAT'
    env = 'Euro'  # visdom 环境

    train_val_ratio = [0.8, 0.2]  # train & val set is fixed
    train_scale = 1  # Scale of training set reduction
    val_scale = 0.2
    # Model related arguments
    sepoch = 1  # use to continue from a checkpoint
    pretrained = False  # use pre-train checkpoint

    # Optimiztion related arguments
    batch_size = 64  # batch size
    max_epochs = None  # 16-[256,256] dataset only need 8~9 epoch
    ckpt_freq = 0
    learning_rate = 1e-3  # initial learning rate
    lr_decay_rate = 0.5
    lr_decay_steps = 20
    lr_scheduler = 'step'  # pre epoch
    warmup = 0  # if warmup > 0, use warmup strategy and end at warmup
    weight_decay = 1e-5  # L2 loss
    optimizer = ['adam', 'sgd', 'lars'][0]
    loss = ['CrossEntropyLoss', ][0]
    loss_weight = None

    # Data related arguments
    num_workers = 4  # number of data loading workers
    dtype = ['RGB']
    bl_dtype = [''][0]
    in_channel = 0

    input_size = (64, 64)  # final input size of network(random-crop use this)
    # crop_params = [256, 256, 256]  # crop_params for val and pre

    # feature_dim = {1: 128, 2: 256, 3: 384, 4: 512}
    mean = [0.5]
    std = [0.5]

    category = EuroSAT()
    classes = category.names
    class_dict = category.table
    num_classes = len(classes)
