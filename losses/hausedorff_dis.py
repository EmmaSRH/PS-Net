#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: hausedorff_dis.py
@time: 2020/7/7 4:46 PM
"""
import cv2
import numba
import numpy as np
import math
from inspect import getmembers
import hausdorff.distances as distances
from numba import cuda

def _find_available_functions(module_name):
    all_members = getmembers(module_name)
    available_functions = [member[0] for member in all_members
                           if isinstance(member[1], numba.core.registry.CPUDispatcher)]
    return available_functions

@numba.jit(nopython=True, fastmath=True)
def _avrg_hausdorff(XA, XB, distance_function, tolerance=1):
    nA = XA.shape[0]
    nB = XB.shape[0]

    c_A = np.zeros(nA)
    count_A = np.zeros(nA)
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
        count_A[i] = cmin
        if cmin <= tolerance:
             cmin = 0
        c_A[i] = cmin
    cavg_A = np.mean(c_A)

    c_B = np.zeros(nB)
    count_B = np.zeros(nB)
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
        count_B[j] = cmin
        if cmin <= tolerance:
            cmin = 0
        c_B[j] = cmin
    cavg_B = np.mean(c_B)
    mean_out = cavg_A + cavg_B

    return mean_out


def hausdorff_distance_with_t(XA, XB, distance='euclidean', tolerance=1):
    assert distance in _find_available_functions(distances), \
        'distance is not an implemented function'
    assert XA.ndim == 2 and XB.ndim == 2, \
        'arrays must be 2-dimensional'
    assert XA.shape[1] == XB.shape[1], \
        'arrays must have equal number of columns'
    distance_function = getattr(distances, distance)
    if XA.shape[0] == 0 and XB.shape[0]==0:
        return 0
    if XA.shape[0] != 0 and XB.shape[0]==0:
        return 1000
    if XA.shape[0] == 0 and XB.shape[0]!=0:
        return 1000
    else:
        return _avrg_hausdorff(XA, XB, distance_function, tolerance=tolerance)

if __name__ == "__main__":
    folder = 'case/1/'
    better_pred = folder + 'better_sk.png'
    bad_pred = folder + 'bad_sk.png'
    gt_path = folder + 'gt_sk.png'

    better = cv2.imread(better_pred, 0)
    bad = cv2.imread(bad_pred, 0)
    gt = cv2.imread(gt_path, 0)

    gt_points = np.argwhere(gt == 0)
    better_points = np.argwhere(better == 0)
    bad_points = np.argwhere(bad == 0)

    better_dis = hausdorff_distance_with_t(gt_points, better_points, distance='euclidean', tolerance=1)
    print(better_dis)