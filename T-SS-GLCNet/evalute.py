# from dltrain import fast_hist
import cv2
# from dldata import get_label_info
import numpy as np
import os
import scipy
# import sklearn
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix as c_matrix
# # path = 'D:/Data_Lib/Seg/MeiTB/1007and2008label'
# label_truth = cv2.imread(path + '/' + '2008test_5label_color.tif')
# label_pred = cv2.imread(path + '/' + '2008_superclass55.tif')
# hist = fast_hist(label_truth, label_pred, 5)

# class_names, label_values = get_label_info(root+'/class_dict.txt')
# class_num = len(class_names)-1


# def class_label(label, label_values):
#     '''
#     Convert RGB label to 2D [HW] array, each pixel value is the classified class key.
#     '''
#     semantic_map = np.zeros(label.shape[:2], label.dtype)
#     for i in range(len(label_values)):
#         equality = np.equal(label, label_values[i])
#         class_map = np.all(equality, axis=-1)
#         semantic_map[class_map] = i
#     return semantic_map


def hist(pred, truth, class_num):
    hist_m = np.zeros([class_num, class_num], dtype=np.int64)
    # for i in range(truth.shape[0]):
    #     for j in range(truth.shape[1]):
    #             col = truth[i][j]
    #             row = pred[i][j]
    #             hist_m[row][col] += 1 
    flat_pred = pred.flatten()
    flat_true = truth.flatten()
    label_class = [x for x in range(class_num)]

    hist_m = c_matrix(flat_true, flat_pred, label_class)
    return hist_m


def matrix(preddataset_path, truthdataset_path, class_Num):
    pred_names = sorted(os.listdir(preddataset_path))
    truth_names = sorted(os.listdir(truthdataset_path))
    matrix1 = np.zeros([class_Num, class_Num], dtype=np.int64)
    for i in range(len(pred_names)):
        pred = cv2.imread(preddataset_path + '/' + pred_names[i])
        truth = cv2.imread(truthdataset_path + '/' + truth_names[i])
        truth[truth == 255] = 0
        # truth[truth == 6] = 1
        # truth[truth == 7] = 2
        # truth[truth == 8] = 3
        # truth[truth == 9] = 4
        matrix1 += hist(pred[:, :, 0], truth[:, :, 0], class_Num)
    return matrix1


def get_scores(hist=None):
    """Returns accuracy score evaluation result.
        - Overall Acc
        - Class Acc
        - Mean Acc
    """
    # hist = self.confusion_matrix if hist is None else hist
    # Overall accuracy
    # hist1 = hist[:, 1:]
    # hist = hist1[1:, :]
    #acc_overall = np.diag(hist)[1:].sum() / (hist.sum()-np.diag(hist)[0]+1e-8)
    acc_overall = np.diag(hist).sum() / (hist.sum()+1e-8)
    # Class accuracy
    acc_cls = np.diag(hist) / (hist.sum(axis=0)+1e-8)  # acc per class
    recall = np.diag(hist) / (hist.sum(axis=1)+1e-8)
    F1_score = (2*acc_cls*recall)/(recall+acc_cls)
    # Class average accuracy
    acc_cls_avg = np.nanmean(acc_cls)
    # Kappa
    n = hist.sum()
    p0 = hist.diagonal().sum()
    p1 = hist.sum(0)
    p2 = hist.sum(1)
    kappa = float(n*p0-np.inner(p1, p2)) / float(n*n - np.inner(p1, p2) + 1e-8)

    # print('\n------Class Acc\n')
    # print(acc_cls)
    # print('\n------recall\n')
    # print(recall)
    # print('\n------F1_score\n')
    # print(F1_score)
    # print('\n------Hist\n')
    # print(hist)
    # print('\n------kappa')
    # print(kappa)
    # print('-----Overall Acc')
    # print(acc_overall)
    # print('-----Mean Acc\n')
    # print(acc_cls_avg)

    return (
        {
            "Hist": hist,  # 混淆矩阵
            "Kappa": kappa,
            "Overall Acc": acc_overall,

            "Class Acc": acc_cls,  # 类别精度
            "recall": recall,
            "F1_score": F1_score,
            "Mean Acc": acc_cls_avg,
        }  # Return as a dictionary
    )


# truth = class_label(label_truth, label_values)
# pred = class_label(label_pred, label_values)


# confusion_matrix
# root = '/project/ytwang/yzw/DeeplabAttASPP/Data'
# # datasetname_list = ['Deeplabv3+_45_pre1','Deeplabv3+_47_pre1','Deeplabv3+_48_pre1_1','Deeplabv3+_48_pre1']
# datasetname_list = ['Deeplabv3+_48_pre1']
# truthset_path = root + 'val_labels'
# class_num = 10
# for i in range(1):
#     # predset_path = root + '/' + datasetname_list[i]
#     confusion_matrix = matrix(predset_path, truthset_path, class_num)
#     print(confusion_matrix)
#     user = np.zeros([1, class_num], dtype=np.int64)
#     prod = np.zeros([1, class_num], dtype=np.int64)
#     for i in range(class_num):
#         for j in range(class_num):
#             a = confusion_matrix[i][j]
#             user[0][i] += a

#             # user = np.zeros([1, 5])
#     for i in range(class_num):
#         for j in range(class_num):
#             prod[0][i] += confusion_matrix[j][i]
#     # p:precision/prod_acc n:recall/user_acc
#     user_acc_recall = np.diag(confusion_matrix)/user
#     prod_acc_presion = np.diag(confusion_matrix)/prod
#     F1_score = (2*prod_acc_presion*user_acc_recall)/(user_acc_recall+prod_acc_presion)
#     print('|||||||||||||||------------------------------------------------------')
#     print('recall/r')
#     print(user_acc_recall)
#     print('class_precision/r')
#     print(prod_acc_presion)
#     print('F1_score/r')
#     print(F1_score)
# #     kappa = get_scores(confusion_matrix)
#     print(kappa)
#     print('------------------------------------------------------||||||||||||||||')