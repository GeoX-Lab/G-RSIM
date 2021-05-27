'''
Tools collection of operations during training.
The icluded functions are:
    def train_log()
    def draw_loss_curve()
    def memory_watcher()
    def Accuracy Evaluation[serval functions]

Version 1.0  2018-04-02 22:44:32
    by QiJi Refence: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
Version 2.0  2018-10-29 09:10:41
    by QiJi
TODO:
    1. 精度评定相关的函数还没Debug

'''
import datetime
import os
import sys

import numpy as np
# from matplotlib import pyplot as plt


# def train_log(X, f=None):
#     ''' Print with time. To console or a file(f) '''
#     time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
#     if not f:
#         sys.stdout.write(time_stamp + " " + X)
#         sys.stdout.flush()
#     else:
#         f.write(time_stamp + " " + X)


# def draw_loss_curve(epochs, loss, path):
#     ''' Paint loss curve '''
#     fig = plt.figure(figsize=(12, 9))
#     ax1 = fig.add_subplot(111)
#     ax1.plot(range(epochs), loss)
#     ax1.set_title("Average loss vs epochs")
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Current loss")
#     plt.savefig(path + '/loss_vs_epochs.png')
#     plt.close(fig)


def memory_watcher():
    ''' Compute the memory usage, for debugging '''
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)


# **********************************************
# *********** Accuracy Evaluation **************
# **********************************************
def compute_global_accuracy(pred, label):
    '''
    Compute the average segmentation accuracy across all classes,
    Input [HW] or [HWC] label
    '''
    count_mat = pred == label
    return np.sum(count_mat) / np.prod(count_mat.shape)


# def compute_class_accuracies(y_pred, y_true, num_classes):
#     ''' Compute the class-specific segmentation accuracy '''
#     # 只能用于计算单张图精度，多张图需要连接处理（计算total时）
#     w = y_true.shape[0]
#     h = y_true.shape[1]
#     flat_image = np.reshape(y_true, w * h)
#     total = []
#     for val in range(num_classes):
#         total.append((flat_image == val).sum())

#     count = [0.0] * num_classes
#     for i in range(w):
#         for j in range(h):
#             if y_pred[i, j] == y_true[i, j]:
#                 count[int(y_pred[i, j])] = count[int(y_pred[i, j])] + 1.0
#     # If there are no pixels from a certain class in the GT, it returns NAN
#     # because of divide by zero, Replace the nans with a 1.0.
#     accuracies = []
#     for i in range(len(total)):
#         if total[i] == 0:
#             accuracies.append(1.0)
#         else:
#             accuracies.append(count[i] / total[i])

#     return accuracies

def compute_class_accuracies(y_pred, y_true, num_classes, total):
    ''' Compute the class-specific segmentation accuracy '''
    # 只能用于计算单张图精度，多张图需要连接处理（计算total时）
    w = y_true.shape[0]
    h = y_true.shape[1]
    # flat_image = np.reshape(y_true, w * h)
    # total = []
    # for val in range(num_classes):
    #     total.append((flat_image == val).sum())

    count = [0.0] * num_classes
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count[int(y_pred[i, j])] = count[int(y_pred[i, j])] + 1.0
    # If there are no pixels from a certain class in the GT, it returns NAN
    # because of divide by zero, Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(0)  # c
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def precision(pred, label):
    '''
    Compute precision
    TODO: Only for 2 class now.
    '''
    TP = np.float(np.count_nonzero(pred * label))
    FP = np.float(np.count_nonzero(pred * (label - 1)))
    prec = TP / (TP + FP)
    return prec


def recall(pred, label):
    '''
    Compute recall.
    TODO: Only for 2 class now.
    '''
    TP = np.float(np.count_nonzero(pred * label))
    FN = np.float(np.count_nonzero((pred - 1) * label))
    rec = TP / (TP + FN)
    return rec


def f1score(pred, label):
    ''' Compute f1 score '''
    prec = precision(pred, label)
    rec = recall(pred, label)
    f1 = np.divide(2 * prec * rec, (prec + rec))
    return f1


def compute_class_iou(pred, gt, num_classes):
    '''
    Args:
        pred: Predict label [HW].
        gt: Ground truth label [HW].
    Return:
        （每一类的）intersection and union list.
    '''
    # If there are no pixels from a certain class in the GT, it returns NAN
    # because of divide by zero,  the class_iou will be 0 ,we
    # pred = pred[:gt.shape[0], :gt.shape[1]]
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for i in range(num_classes):
        pred_i = pred == i
        label_i = gt == i
        # label_i.resize(pred.shape[0], pred.shape[1]) #
        pred_i = pred_i[:label_i.shape[0], :label_i.shape[1]]
        intersection[i] = float(np.sum(np.logical_and(label_i, pred_i)))
        union[i] = float(np.sum(np.logical_or(label_i, pred_i)) + 1e-8)
    class_iou = intersection / union
    return class_iou


def evaluate_segmentation(pred, gt, num_classes):
    '''
    Evaluate Segmentation result

    Args:
        pred: Predict label [HW].
        gt: Ground truth label [HW].
        num_classes: Num of classes.
    Returns:
        accuracy:
        class_accuracies:
        prec:
        rec:
        f1:
        iou:
    '''
    accuracy = compute_global_accuracy(pred, gt)
    class_accuracies = compute_class_accuracies(pred, gt, num_classes)
    prec = precision(pred, gt)
    rec = recall(pred, gt)
    f1 = f1score(pred, gt)
    i, u = compute_class_iou(pred, gt, num_classes)
    iou = np.mean(i / u)
    return accuracy, class_accuracies, prec, rec, f1, iou


# **********************************************
# ***********                     **************
# **********************************************
