#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: hausdorff_dis2.py.py
@time: 2020/9/19 6:12 PM
"""
import cv2
import numba
import numpy as np
import math
from inspect import getmembers
import hausdorff.distances as distances


def _find_available_functions(module_name):
    all_members = getmembers(module_name)
    available_functions = [member[0] for member in all_members
                           if isinstance(member[1], numba.core.registry.CPUDispatcher)]
    return available_functions

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))

@numba.jit(nopython=True, fastmath=True)
def _avrg_hausdorff(XA, XB, distance_function):
    nA = XA.shape[0]
    nB = XB.shape[0]

    c_A = np.zeros(nA)
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
        c_A[i] = cmin
    csum_A = np.sum(c_A)

    c_B = np.zeros(nB)
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
        c_B[j] = cmin

    csum_B = np.sum(c_B)

    return (csum_A+csum_B)/(nA+nB)


def get_assd(XA, XB, distance='euclidean'):
    assert distance in _find_available_functions(distances), \
        'distance is not an implemented function'
    assert XA.ndim == 2 and XB.ndim == 2, \
        'arrays must be 2-dimensional'
    assert XA.shape[1] == XB.shape[1], \
        'arrays must have equal number of columns'
    if XA.shape[0] == 0 and XB.shape[0] == 0:
        return 0
    else:
        distance_function = getattr(distances, distance)
        return _avrg_hausdorff(XA, XB, distance_function)


if __name__ == "__main__":
    pred_path= '/Users/shiwakaga/Desktop/Experiments/segmentation_skeletons/0141_1_1565791505_66/7168_8192_1024_2048/A.png'
    gt_path = '/Users/shiwakaga/Desktop/Experiments/segmentation_skeletons/0141_1_1565791505_66/7168_8192_1024_2048/gt.png'


    pred = cv2.imread(pred_path, 0)
    gt = cv2.imread(gt_path, 0)

    gt_points = np.argwhere(gt == 0)
    better_points = np.argwhere(pred == 0)


    score = get_assd(gt_points, better_points, distance='euclidean')
    print(score)
