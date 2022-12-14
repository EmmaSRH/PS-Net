#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: hausedorff_dis.py.py
@time: 2020/7/7 4:46 PM
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

@numba.jit(nopython=True, fastmath=True)
def _hausdorff(XA, XB, distance_function):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0.
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(XA[i,:], XB[j,:])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = distance_function(XA[i,:], XB[j,:])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    return cmax


def hausdorff_distance(XA, XB, distance='euclidean'):
    assert type(XA) is np.ndarray and type(XB) is np.ndarray, \
        'arrays must be of type numpy.ndarray'
    assert np.issubdtype(XA.dtype, np.number) and np.issubdtype(XA.dtype, np.number), \
        'the arrays data type must be numeric'
    assert XA.ndim == 2 and XB.ndim == 2, \
        'arrays must be 2-dimensional'
    assert XA.shape[1] == XB.shape[1], \
        'arrays must have equal number of columns'

    if isinstance(distance, str):
        assert distance in _find_available_functions(distances), \
            'distance is not an implemented function'
        if distance == 'haversine':
            assert XA.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
            assert XB.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
        distance_function = getattr(distances, distance)
    elif callable(distance):
        distance_function = distance
    else:
        raise ValueError("Invalid input value for 'distance' parameter.")
    return _hausdorff(XA, XB, distance_function)


if __name__ == "__main__":
    # folder = 'case/1/'
    # gt_path = '/mnt/srh/U-RISC-DATASET/label/complex/test/0128_1_1565791505_98.png'
    # pre_path = '/home/srh/Add_PHD_loss/predictions/prediction-10-13/U_Net/U_Net-399_pre_ful/0128_1_1565791505_98.png'
    #
    # gt_points = np.argwhere(cv2.imread(gt_path,0))
    # pre_points = np.argwhere(cv2.imread(pre_path,0))
    # print(len(gt_points))
    # print(len(pre_points))
    #
    # phd_dis = hausdorff_distance(gt_points, pre_points, distance='euclidean')
    # print(phd_dis)

    gt = np.argwhere(cv2.imread('/Users/shiwakaga/Desktop/gt_sk.png', 0) == 0)
    pre = np.argwhere(cv2.imread('/Users/shiwakaga/Desktop/pre_sk.png', 0) == 0)
    phd_dis = hausdorff_distance(gt, pre, distance='euclidean')
    print(phd_dis)
