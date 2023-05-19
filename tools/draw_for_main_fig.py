#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: draw_for_main_fig.py
@time: 2021/10/27 2:58 PM
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import medial_axis, skeletonize

if __name__ == '__main__':
    ori_img = cv2.imread('/Users/shiwakaga/PyTorch-ISBI2012-Segmentation/ISBI2012/imgs/test/01.png',0)[:128,:128]
    ori_label = cv2.imread('/Users/shiwakaga/PyTorch-ISBI2012-Segmentation/ISBI2012/labels/test/01.png',0)[:128,:128]


    plt.figure(1)
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(ori_img, cmap='gray')
    # ax1.set_title('EM Image')
    ax1.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(ori_label)
    # ax2.set_title('Membrane Annotation')
    ax2.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    _, binary = cv2.threshold(ori_label, 0.5, 1, cv2.THRESH_BINARY_INV)
    binary[binary == 1] = 1
    skeleton = skeletonize(binary)
    skel, distance = medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    sk = 255 - skeleton.astype(np.uint8) * 255

    ax3.imshow(dist_on_skel, cmap='magma')
    ax3.contour(binary, [0.5], colors='w')
    # ax3.set_title('Topology Extraction')
    ax3.axis('off')

    # ax2.imshow(ori_label)


    plt.savefig('main_fig1.pdf')
    # plt.show()


    # plt.figure(1)
    # ax1 = plt.subplot(1, 1, 1)
    # img = cv2.imread('/Users/shiwakaga/Desktop/test_1.png', 0)
    # _, binary = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY_INV)
    # binary[binary == 1] = 1
    # skeleton = skeletonize(binary)
    # skel, distance = medial_axis(binary, return_distance=True)
    # dist_on_skel = 255-distance * skel
    # sk = 255-skeleton.astype(np.uint8) * 255
    #
    # ax1.imshow(dist_on_skel, cmap='magma')
    # ax1.contour(binary, [0.5], colors='b')
    # ax1.axis('off')
    # plt.savefig('T_method.png')



