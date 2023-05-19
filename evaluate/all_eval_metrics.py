#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: all_eval_metrics.py
@time: 2021/10/17 12:59 AM
"""
import cv2
import numpy as np
from hausedorff_dis import hausdorff_distance_with_t
from assd import get_assd
from hausedorff_dis_original import hausdorff_distance
# from vrand_vinfo import evl as evl_vrand_info
from skimage import morphology
import csv
import glob


def compute_iou(gt, pre):
    # load label
    image_mask = 1 - gt/ 255
    predict = 1 - pre / 255

    # compute intersection areas between prediction and label
    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea  # compute union areas between prediction and label
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union  # IOU = intersection / union

    return iou_tem

def single_eval_boundary(fg_boundary, gt_boundary, bound_pix=0):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        fg_boundary (ndarray): binary boundary prediction image.     Shape: H x W     Value: 0 and 255
        gt_boundary (ndarray): binary annotated boundary image.      Shape: H x W     Value: 0 and 255
        bound_pix: half of the thickness of boundary.
                  A morphology dilation will make the thickness of edge to what we set here
                   this is the radius of disk

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(fg_boundary).shape[2] == 1

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F meas
    # ure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F, precision, recall

def eval_on_onepic(pred, gt):
    pred_ = 255 - pred
    gt_ = 255 - gt

    gt_[gt_ < 127.5] = 0
    gt_[gt_ >= 127.5] = 255

    # check_if_white_back_black_edge(pred_)

    F1, precision, recall = single_eval_boundary(pred_, gt_, bound_pix=0)
    return F1, precision, recall


def TNVF(pre,gt):
    """

    :param pre:
    :param gt:
    :return: tn/fp+tn
    """
    pred_ = 255 - pre
    gt_ = 255 - gt

    gt_[gt_ < 122.5] = 0
    gt_[gt_ >= 122.5] = 1

    pred_[pred_ < 122.5] = 0
    pred_[pred_ >= 122.5] = 1

    intersection = (pred_ * gt_).sum()
    all = pred_.shape[0]*pred_.shape[1]
    unionArea = pred_.sum() + gt_.sum() - intersection

    score = (all-unionArea)/(all-gt_.sum())

    return score


def RVD(pre,gt):
    pred_ = 255 - pre
    gt_ = 255 - gt

    gt_[gt_ < 127.5] = 0
    gt_[gt_ >= 127.5] = 1

    pred_[pred_ < 127.5] = 0
    pred_[pred_ >= 127.5] = 1

    if pred_.sum()==0 and gt_.sum()==0:
        return 1
    if pred_.sum()!=0 and gt_.sum()==0:
        return 0
    else:
        diff = np.abs(pred_.sum()-gt_.sum())
        score = diff / gt.sum()

        return score

def get_ASSD_scores(pre,gt):

    gt_points = np.argwhere(gt == 0)
    pre_points = np.argwhere(pre == 0)

    score = get_assd(gt_points, pre_points)

    return score

def Hausdorff(pre,gt):
    """
    get hd score for gt and pre image
    :param pre: 2d array
    :param gt: 2d array
    :return: score number
    """
    gt_points = np.argwhere(gt == 0)
    pre_points = np.argwhere(pre == 0)

    score = hausdorff_distance(gt_points, pre_points)

    return score

def assd_evl(pre,gt):
    """
        get assd score for gt and pre image
        :param pre: 2d array
        :param gt: 2d array
        :return: score number
        """
    gt_points = np.argwhere(gt == 0)
    pre_points = np.argwhere(pre == 0)

    score = get_assd(gt_points, pre_points)

    return score


def get_points(arr):
    _, binary = cv2.threshold(arr, 0.5, 1, cv2.THRESH_BINARY_INV)
    binary[binary == 1] = 1
    skeleton = morphology.skeletonize(binary)
    points = np.argwhere(skeleton == True)
    return points

if __name__ == '__main__':
    # gt_paths = glob.glob('/mnt/srh/U-RISC-DATASET/label/complex/test/*.png')
    # pre_paths = '/home/srh/Add_PHD_loss/predictions/prediction-10-14/U_Net/U_Net-599_pre_ful/'
    # pre_fol_name =pre_paths.split('/')[-4]
    # pre_model_name = pre_paths.split('/')[-2].split('-')[1].split('_')[0]
    # with open(pre_fol_name +'-'+pre_model_name+'-metrics.csv','a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['img_name', 'f1', 'precision','recall','iou','hd', 'tnvf','rvd','hd-sk', 'assd','assd-sk', 'phd0','phd1', 'phd3', 'phd5', 'phd10'])
    #     for i in range(len(gt_paths)):
    #         # read img as binary image
    #         img_name = gt_paths[i].split('/')[-1].split('.')[0]
    #         gt_path = gt_paths[i]
    #         pre_path = pre_paths + img_name + '.png'
    #         print(gt_path,pre_path)
    #
    #         pred_ = cv2.imread(pre_path, 0)
    #         gt_ = cv2.imread(gt_path, 0)
    #
    #         # evaluate F1 score
    #         bound_pix = 0  # the thickness of boundary
    #         thread_num = 4  # number of parallel threads
    #         F1, precision, recall = eval_on_onepic(pred_, gt_)
    #
    #         # evaluate iou score
    #         iou = compute_iou(gt_, pred_)
    #
    #         # evaluate iou score
    #         hd = Hausdorff(pred_, gt_)
    #
    #         # evaluate tnvf score
    #         tnvf = TNVF(pred_, gt_)
    #
    #         # evaluate rvd score
    #         rvd = RVD(pred_, gt_)
    #
    #         print('img_name',img_name)
    #         print('F1', F1)
    #         print('precision',precision)
    #         print('recall',recall)
    #         print('iou',iou)
    #         print('hd',hd)
    #         print('tnvf',tnvf)
    #         print('rvd',rvd)
    #
    #
    #         # evaluate PHD score
    #         gt_points = get_points(gt_)
    #         pre_points = get_points(pred_)
    #
    #         # evaluate iou score
    #         hd_sk = hausdorff_distance(gt_points, pre_points)
    #
    #         # evaluate assd score
    #         assd = assd_evl(pred_, gt_)
    #         assd_sk = get_assd(gt_points, pre_points)
    #
    #
    #         # tolerance = 1  # choose an int number
    #         PHD_0 = hausdorff_distance_with_t(gt_points, pre_points, distance='euclidean', tolerance=0)
    #         PHD_1 = hausdorff_distance_with_t(gt_points, pre_points, distance='euclidean', tolerance=1)
    #         PHD_3 = hausdorff_distance_with_t(gt_points, pre_points, distance='euclidean', tolerance=3)
    #         PHD_5 = hausdorff_distance_with_t(gt_points, pre_points, distance='euclidean', tolerance=5)
    #         PHD_10 = hausdorff_distance_with_t(gt_points, pre_points, distance='euclidean', tolerance=10)
    #
    #
    #         print('hd_sk',hd_sk)
    #         print('assd',assd)
    #         print('assd_sk',assd_sk)
    #         print('PHD_0',PHD_0)
    #         print('PHD_1',PHD_1)
    #         print('PHD_3',PHD_3)
    #         print('PHD_5',PHD_5)
    #         print('PHD_10',PHD_10)
    #
    #         writer.writerow(
    #             [img_name, F1, precision, recall, iou, hd, tnvf, rvd, hd_sk, assd, assd_sk, PHD_0, PHD_1, PHD_3, PHD_5, PHD_10])
    #
    #

    pre1 = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/A.png',0)
    pre2 = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/B.png',0)
    pre3 = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/C.png',0)
    pre4 = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/D.png',0)
    pre5 = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/E.png',0)
    pre6 = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/F.png',0)
    gt = cv2.imread('/Users/shiwakaga/PKU/iclr2021_experiments/Experiments/Test_imgs/0078_1_1565791505_0/8935_9959_0_1024/gt.png',0)

    F1, _,_ = eval_on_onepic(pre1, gt)
    iou =  compute_iou(gt, pre1)
    print(F1,iou)
    F1, _,_ = eval_on_onepic(pre2, gt)
    iou =  compute_iou(gt, pre2)
    print(F1,iou)
    F1, _,_ = eval_on_onepic(pre3, gt)
    iou =  compute_iou(gt, pre3)
    print(F1,iou)
    F1, _,_ = eval_on_onepic(pre4, gt)
    iou =  compute_iou(gt, pre4)
    print(F1,iou)
    F1, _,_ = eval_on_onepic(pre5, gt)
    iou =  compute_iou(gt, pre5)
    print(F1,iou)
    F1, _,_ = eval_on_onepic(pre6, gt)
    iou =  compute_iou(gt, pre6)
    print(F1,iou)
    # # evaluate vrand, vinfo score
    # vrand, vinfo = evl_vrand_info(pre_path, gt_path)
    #





