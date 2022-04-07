#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about CVPR_LandCover dataset, segmentation mode
'''
from config.category import CVPR_LandCover


class Config(object):
    ''' Fixed initialization Settings '''
    # Path and file
    root = ''
    data_dir = root + 'segmentation/Seg_data/CVPR_LandCover'
    ckpt = root + 'Task_models/CVPR_LandCover'
    env = 'CVPRLC'  # visdom 环境

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
    # crop_params = [256, 256, 256]  # crop_params for val and pre

    # feature_dim = {1: 128, 2: 256, 3: 384, 4: 512}
    mean = [0.5, 0.5, 0.5]  # BGR, 此处的均值应该是0-1
    std = [0.5, 0.5, 0.5]  # [0.12283102, 0.1269429, 0.15580289]

    category = CVPR_LandCover()
    classes = category.names
    class_dict = category.table
    num_classes = len(classes)

    reduce_zero_label = False
    ignore_index = None
