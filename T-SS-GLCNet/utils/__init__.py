# -*- coding:utf-8 -*-
'''
Abstract.

Version 1.0  2019-12-06 22:17:13 by QiJi
TODO:
1. 目前只改了DinkNet50的 first_conv，其余的都还得修改

'''
from .import data_for_self_inpaitting,data_for_self_contrast,data_for_self_Jigsaw,data_for_self_GLCNet

#from torchvision import models


def get_self_dataset(args):
    '''
    Args:
        net_num: 6位数,
        第1位代表main model的arch:
        0 - Scene_Base
            第2位代表Backbone 的arch：
            1 - resnet34
            2 - resnet50
            3 - resnet101
            4 - vgg16
            5 - googlenet
    '''
    if args.self_mode==1 or args.self_mode==11 or args.self_mode==12 or args.self_mode==13:

        train_dataset = data_for_self_GLCNet.Train_Dataset(args.dataset_dir, args.self_data_name, args)
    elif args.self_mode==2 or args.self_mode==3:
            train_dataset = data_for_self_contrast.Train_Dataset(args.dataset_dir, args.self_data_name, args)

    elif args.self_mode==4:
        train_dataset = data_for_self_inpaitting.Train_Dataset(args.dataset_dir, args.self_data_name, args)

    elif args.self_mode==5:
        train_dataset = data_for_self_Jigsaw.Train_Dataset(args.dataset_dir, args.self_data_name, args)



    return train_dataset
