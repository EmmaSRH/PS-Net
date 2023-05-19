#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: eval.py.py
@time: 2021/2/22 7:45 PM
"""

from evaluation import *
from F1_eval import *
import glob

if __name__ == '__main__':
    gt_path = '/Users/shiwakaga/Downloads/U-RISC-DATASET/label/complex/test'
    pre_path = '/home/shiruohua/Pytorch-Topology-Aware-Delineation/Topology-Aware-Delineation/prediction/060_full'


    bound_pix = 0  # the thickness of boundary
    thread_num = 4  # number of parallel threads

    mean_F = eval_on_whole_dataset(pre_path, gt_path, bound_pix, thread_num)


