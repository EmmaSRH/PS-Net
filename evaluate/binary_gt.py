#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: binary_gt.py
@time: 2021/11/11 7:31 PM
"""
import cv2

import glob

gts = glob.glob('/Users/shiwakaga/Downloads/U-RISC-DATASET/label/simple/*/*.tiff') + glob.glob('/Users/shiwakaga/Downloads/U-RISC-DATASET/label/simple/*/*.png')
for gt in gts:
    print(gt)
    img = cv2.imread(gt,0)
    img[img>127.5]=255
    img[img<=127.5]=0
    cv2.imwrite(gt.replace('Downloads/U-RISC-DATASET/','Desktop/challenge/').replace('.tiff','.png'),img)