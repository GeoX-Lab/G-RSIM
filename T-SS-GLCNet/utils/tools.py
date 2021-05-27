'''
Road extration expriment tools.

by QiJi
TODO:
1. xxx

'''
# import os
import cv2
import torch
import numpy as np
# from tqdm import tqdm
from dl_tools.basictools.dldata import vote_combine
# import re


def net_predict1(net, image, opt, crop_info=None):
    ''' Do predict use Net(only one image at a time). '''
    if crop_info is None:
        # predict a complete image
        if opt.use_gpu:
                image = image.cuda()
        output = net(image)[0]  # [NCHW] -> [CHW]
        predict = np.argmax(output.cpu().detach().numpy(), 0)  # [CHW] -> [HW]
    else:
        # predict the list of croped images
        predict = []
        image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for input in image:
            if opt.use_gpu:
                input = input.cuda()  # [NCHW](N=1)
            output = net(input)[0]  # [NCHW] -> [CHW]
            # output = output[0]
            # tmp = np.mean(input, axis=1, dtype=input.type)
            # indx = np.argwhere(tmp == 0)
            # output[indx] = 0
            # output = output[0]
            output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))  # [CHW]->[HWC]
            if opt.input_size != opt.crop_params[:2]:
                    output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
            predict.append(output)
        predict = vote_combine(predict, opt.crop_params, crop_info, 2)
        predict = np.argmax(predict, -1)  # [HWC] -> [HW]

    return predict  # [HW] array


def net_predict2(net, image, opt, crop_info=None):
    ''' Do predict use Net(only one image at a time). '''
    if crop_info is None:
        # predict a complete image
        if opt.use_gpu:
                image = image.cuda()
        output = net(image)[0]  # [NCHW] -> [CHW]
        predict = np.argmax(output.cpu().detach().numpy(), 0)  # [CHW] -> [HW]
    else:
        # predict the list of croped images
        predict = []
        image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for input in image:
            if opt.use_gpu:
                input = input.cuda()  # [NCHW](N=1)
            output = net(input)[0]  # [NCHW] -> [CHW]
            output = output[0]
            # tmp = np.mean(input, axis=1, dtype=input.type)
            # indx = np.argwhere(tmp == 0)
            # output[indx] = 0
            # output = output[0]
            output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))  # [CHW]->[HWC]
            if opt.input_size != opt.crop_params[:2]:
                    output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
            predict.append(output)
        predict = vote_combine(predict, opt.crop_params, crop_info, 2)
        predict = np.argmax(predict, -1)  # [HWC] -> [HW]

    return predict  # [HW] array


def net_predict_enhance(net, image, opt, crop_info=None):
    ''' Do predict use Net with some trick(only one image at a time). '''
    predict_list = []
    if crop_info is None:
        # predict a complete image
        for i in range(4):
            input = torch.from_numpy(np.rot90(image, i, axes=(3, 2)).copy())
            if opt.use_gpu:
                input = input.cuda()
            output = net(input)[0]  # [NCHW] -> [CHW]
            output = output.cpu().detach().numpy()  # Tensor -> array
            output = np.transpose(output, (1, 2, 0))  # [CHW]->[HWC]
            predict_list.append(np.rot90(output, i, axes=(0, 1)))  # counter-clockwise rotation

    else:
        # predict the list of croped images
        image.permute(1,0,2,3,4)#image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for i in range(4):
            predict = []
            for img in image:
                input = torch.from_numpy(np.rot90(img, i, axes=(3, 2)).copy())
                if opt.use_gpu:
                    input = input.cuda()  # [NCHW](N=1)
                output = net(input)[0]  # [NCHW] -> [CHW]
                output = output.cpu().detach().numpy()
                output = np.transpose(output, (1, 2, 0))  # [CHW]->[HWC]
                if opt.input_size != opt.crop_params[:2]:
                    output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
                predict.append(np.rot90(output, i, axes=(0, 1)))
            predict_list.append(vote_combine(predict, opt.crop_params, crop_info, 2))

    predict = predict_list[0]
    for i in range(1, 4):
        predict += predict_list[i]
    return np.argmax(predict, -1)  # [HWC] -> [HW] array


def net_predict_enhance2(net, image, opt, crop_info=None):
    ''' Do predict use Net with some trick(only one image at a time). '''
    predict_list = []
    if crop_info is None:
        # predict a complete image
        for i in range(4):
            input = torch.from_numpy(np.rot90(image, i, axes=(3, 2)).copy())
            if opt.use_gpu:
                input = input.cuda()
            output = net(input)[0]  # [NCHW] -> [CHW]
            output = output.cpu().detach().numpy()  # Tensor -> array
            output = np.transpose(output, (1, 2, 0))  # [CHW]->[HWC]
            predict_list.append(np.rot90(output, i, axes=(0, 1)))  # counter-clockwise rotation

    else:
        # predict the list of croped images
        image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for i in range(4):
            predict = []
            for img in image:
                input = torch.from_numpy(np.rot90(img, i, axes=(3, 2)).copy())
                if opt.use_gpu:
                    input = input.cuda()  # [NCHW](N=1)
                output = net(input)[0]  # [NCHW] -> [CHW]
                output = output.cpu().detach().numpy()
                output = np.transpose(output, (1, 2, 0))  # [CHW]->[HWC]
                if opt.input_size != opt.crop_params[:2]:
                    output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
                predict.append(np.rot90(output, i, axes=(0, 1)))
            predict_list.append(vote_combine(predict, opt.crop_params, crop_info, 2))

    predict = predict_list[0]
    for i in range(1, 4):
        predict += predict_list[i]
    return np.argmax(predict, -1)  # [HWC] -> [HW] array
