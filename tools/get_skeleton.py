#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: get_skeleton.py
@time: 2021/8/4 5:26 PM
"""
import cv2
from skimage import morphology
import numpy as np
import time
import glob
import os

all_gts = glob.glob('/mnt/srh/U-RISC-DATASET/patchs/2048/test/*/*.jpg')

for gt in all_gts:
    print(gt)
    filename1 = gt.split('/')[-2]
    filename2 = gt.split('/')[-1]
    save_root = '/mnt/srh/U-RISC-DATASET/patchs/2048/test_sk/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_dir = save_root + filename1 + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = cv2.imread(gt,0)
    _,binary = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    binary[binary == 255] = 1

    start1 = time.time()
    skeleton0 = morphology.skeletonize(binary)
    end1 = time.time()
    print(end1 - start1) # 0.039054155349731445 time long but fine
    skeleton = skeleton0.astype(np.uint8)*255
    cv2.imwrite(save_dir + filename2.replace('.jpg','.png'),skeleton)



    # start2 = time.time()
    # skel, distance =morphology.medial_axis(binary, return_distance=True)
    # end2 = time.time()
    # print(end2 - start2) # 0.30717897415161133 time short but not good
    # dist_on_skel = distance * skel
    # dist_on_skel = dist_on_skel.astype(np.uint8)*255
    # cv2.imwrite("dist_on_skel.png",dist_on_skel)