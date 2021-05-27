'''
深度学习基础工具
'''
# import sys
# import os
# import cv2
import numpy as np
# from scipy import misc
# import matplotlib.pyplot as plt
# import re


# **********************************************
# **********  Result visualization  ************
# **********************************************
# def visualize_feature_map(FM, outdir, num=1, name=None):
#     '''
#     Parse the input FeatureMap and do sample visualization
#     Args:
#         input: Tensor of feature map.
#         outdir: Dir of ouput image saving.

#     '''
#     FM = np.array(FM)
#     if len(FM.shape) == 5:  # [11HWC] -> [1HWC]
#         FM = FM[0]
#     if len(FM.shape) == 4:  # [1HWC] -> [HWC]
#         FM = FM[0]

#     h, w, c = FM.shape
#     # 归一化 0-1
#     tp_min, tp_max = np.min(FM), np.max(FM)
#     if tp_min < 0:
#         FM += abs(tp_min)
#     FM /= (tp_max - tp_min)

#     # Iterate filters
#     for i in range(c):
#         fig = plt.figure(figsize=(12, 12))
#         axes = fig.add_subplot(111)
#         img = FM[:, :, i]
#         # Toning color
#         axes.imshow(img, vmin=0, vmax=0.9, interpolation='bicubic', cmap='coolwarm')
#         # Remove any labels from the axes
#         # axes.set_xticks([])
#         # axes.set_yticks([])

#         # Save figure
#         if name is None:
#             # misc.imsave(outdir + '/%d_%.3d.png' % (num, i), FM[:, :, i])
#             plt.savefig(outdir + '/%d_%.3d.png' % (num, i), dpi=60, bbox_inches='tight')
#         else:
#             # misc.imsave(outdir + '/%s_%.3d.png' % (name, i), FM[:, :, i])
#             plt.savefig(outdir + '/%s_%.3d.png' % (name, i), dpi=60, bbox_inches='tight')
#         plt.close(fig)


if __name__ == '__main__':
    # in_dir = '/home/tao/Data/VOCdevkit2007/VOC2007/small/train'
    # rename_files(in_dir)
    # mkdir_of_dataset('/home/tao/Data/XM')
    pass
