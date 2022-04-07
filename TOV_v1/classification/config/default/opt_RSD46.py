#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about WHU_RSD46 dataset, classification mode
'''
from config.category import WHU_RSD46


class Config(object):
    ''' Fixed initialization Settings '''
    # Path and file
    root = ''
    data_dir = root + 'classification/Cls_data/WHU_RSD46'
    # data_dir = '/dev/shm/WHU_RSD46'
    ckpt = root + 'Task_models/WHU_RSD46'
    env = 'RSD46'  # visdom 环境

    train_val_ratio = [0, 0]  # train & val set is fixed
    train_scale = 1  # Scale of training set reduction
    val_scale = 0.1
    # Model related arguments
    sepoch = 1  # use to continue from a checkpoint
    pretrained = False  # use pre-train checkpoint

    # Optimiztion related arguments
    batch_size = 16  # batch size
    max_epochs = None  # 16-[256,256] dataset only need 8~9 epoch
    ckpt_freq = 10
    learning_rate = 4e-4  # initial learning rate
    lr_scheduler = 'step'  # pre epoch
    lr_decay_rate = 0.5
    lr_decay_steps = 20
    warmup = 0  # if warmup > 0, use warmup strategy and end at warmup
    weight_decay = 1e-5  # L2 loss
    optimizer = ['adam', 'sgd', 'lars'][0]
    loss = ['CrossEntropyLoss', 'NTXentloss'][0]
    loss_weight = None

    # Data related arguments
    num_workers = 4  # number of data loading workers
    dtype = ['RGB'][0]
    bl_dtype = [''][0]
    in_channel = 0

    input_size = (224, 224)  # final input size of network(random-crop use this)
    # crop_params = [256, 256, 256]  # crop_params for val and pre

    # feature_dim = {1: 128, 2: 256, 3: 384, 4: 512}
    mean = [0.5, 0.5, 0.5]  # BGR, 此处的均值应该是0-1
    std = [0.5, 0.5, 0.5]  # [0.12283102, 0.1269429, 0.15580289]

    category = WHU_RSD46()
    classes = category.names
    class_dict = category.table
    num_classes = len(classes)
