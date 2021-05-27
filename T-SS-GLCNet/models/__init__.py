# -*- coding:utf-8 -*-
'''
Abstract.

Version 1.0  2019-12-06 22:17:13 by QiJi
TODO:
1. 目前只改了DinkNet50的 first_conv，其余的都还得修改

'''
from .import v3p,v3p_SimCLR_encoder,GLCNet,v3p_inpaiting_encoder,v3p_jigsaw_encoder,v3p_mocov2_encoder_true


#from torchvision import models


def build_model(args):
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
    if args.self_mode==1:
        model=GLCNet.build_model(num_classes=64, in_channels=args.n_channels, pretrained=False, arch=args.arch, patch_num=args.patch_num,
                        patch_size=args.patch_size, patch_out_channel=False, pross_num=args.pross_num)  # pretrained=False,
    elif args.self_mode == 11:
        model = GLCNet.build_model(num_classes=64, in_channels=args.n_channels, pretrained=False, arch=args.arch,
                                   patch_num=args.patch_num,
                                   patch_size=args.patch_size, patch_out_channel=False,noStyle=True,
                                   pross_num=args.pross_num)  # pretrained=False,
    elif args.self_mode == 12:
        model = GLCNet.build_model(num_classes=64, in_channels=args.n_channels, pretrained=False, arch=args.arch,
                                   patch_num=args.patch_num,
                                   patch_size=args.patch_size, patch_out_channel=False,noGlobal=True,
                                   pross_num=args.pross_num)  # pretrained=False,

    elif args.self_mode == 13:
        model = GLCNet.build_model(num_classes=64, in_channels=args.n_channels, pretrained=False, arch=args.arch,
                                   patch_num=args.patch_num,
                                   patch_size=args.patch_size, patch_out_channel=False,noLocal=True,
                                   pross_num=args.pross_num)  # pretrained=False,
    elif args.self_mode==2:
        model = v3p_SimCLR_encoder.build_model(num_classes=64, in_channels=args.n_channels, pretrained=False,
                                   arch=args.arch)  # pretrained=False,
    elif args.self_mode==3:
        m = 0.999
        K = 2048
        model = v3p_mocov2_encoder_true.build_model(num_classes=64, in_channels=args.n_channels, pretrained=False,
                                                    arch=args.arch, m=m,
                                                    K=K)  # pretrained=False,
    elif args.self_mode==4:
        model = v3p_inpaiting_encoder.build_model(in_channels=args.n_channels, pretrained=False,
                                                  arch=args.arch)  # pretrained=False,
    elif args.self_mode==5:
        model=v3p_jigsaw_encoder.build_model(num_classes=1000, in_channels=args.n_channels, pretrained=False,
                                       arch=args.arch)
    elif args.self_mode==0:
        model = v3p.build_model(num_classes=args.class_num, in_channels=args.n_channels, pretrained=False,
                              arch=args.arch)  # num_classes1=50,

    if hasattr(model, 'model_name'):
        print(model.model_name)
    return model
