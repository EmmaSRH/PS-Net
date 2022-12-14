#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: count_distance.py
@time: 2021/10/26 8:49 PM
"""
import glob
import csv
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
from skimage import morphology
# import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns

def _find_available_functions(module_name):
    all_members = getmembers(module_name)
    available_functions = [member[0] for member in all_members
                           if isinstance(member[1], numba.core.registry.CPUDispatcher)]
    return available_functions

@numba.jit(nopython=True, fastmath=True)
def _avrg_hausdorff(XA, XB, distance_function):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0.
    c_A = []
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
        c_A.append(cmin)
    c_B = []
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
        c_B.append(cmin)
    return c_A, c_B


def hausdorff_distance_with_t(XA, XB, distance='euclidean'):
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
        return 0
    if XA.shape[0] == 0 and XB.shape[0]!=0:
        return 0
    else:
        return _avrg_hausdorff(XA, XB, distance_function)

def get_points(arr):
    _, binary = cv2.threshold(arr, 0.5, 1, cv2.THRESH_BINARY_INV)
    binary[binary == 1] = 1
    skeleton = morphology.skeletonize(binary)
    points = np.argwhere(skeleton == True)
    return points

def get_sk_img(arr):
    _, binary = cv2.threshold(arr, 0.5, 1, cv2.THRESH_BINARY_INV)
    binary[binary == 1] = 1
    skeleton = morphology.skeletonize(binary)
    return 255-skeleton.astype(np.uint8) * 255

def gey_all_sks():
    gt_paths = glob.glob('/mnt/shiruohua/U-RISC-DATASET/label/complex/test/*.png')
    pre_paths = '/home/srh/Add_PHD_loss/predictions/prediction-10-13/U_Net/U_Net-399_pre_ful/'
    save_path = '/home/srh/Add_PHD_loss/cvpr2022/'
    for gt_path in gt_paths:
        print(gt_path)
        img_name = gt_path.split('/')[-1]
        gt = cv2.imread(gt_path, 0)
        pre = cv2.imread(pre_paths + img_name, 0)
        cv2.imwrite(save_path+'gt/'+img_name, get_sk_img((255 - gt) / 255))
        cv2.imwrite(save_path+'pre/'+img_name, get_sk_img((255 - pre) / 255))
    return
def count_dis():
    gt_paths = glob.glob('/home/srh/Add_PHD_loss/cvpr2022/gt/*.png')
    pre_paths = '/home/srh/Add_PHD_loss/cvpr2022/pre/'
    save_path = '/home/srh/Add_PHD_loss/cvpr2022/'
    for gt_path in gt_paths:
        print(gt_path)
        img_name = gt_path.split('/')[-1]
        gt = cv2.imread(gt_path, 0)
        pre = cv2.imread(pre_paths + img_name, 0)
        with open('{}.csv'.format(img_name), 'a') as f:
            writer = csv.writer(f)
            gt_pts = np.argwhere(gt == 0)
            pre_pts = np.argwhere(pre == 0)
            c_A, c_B = hausdorff_distance_with_t(gt_pts, pre_pts, distance='euclidean') # c_A, c_B
            writer.writerow(c_A)
            writer.writerow(c_B)
    return

def analysis_dis():
    # # analysis on dis
    with open('dis_count_res.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ['file_name', 'num_pts', 'max', 'min', ',mean', '<5', '<10', '<100', '<200', '<250', '<300', '<350', '<400',
             '<500', '<1000', '<2000'])
        csv_files = glob.glob('/Users/shiwakaga/Desktop/dis_count/*.csv')
        for csv_file in csv_files:
            file_name = csv_file.split('/')[-1]
            print(file_name)
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                i = 0
                for row in reader:
                    i += 1
                    c_gt = np.array([float(x) for x in row])
                    # c_pre = reader[1]
                    print(len(c_gt))
                    print(np.max(c_gt))
                    print(np.min(c_gt))
                    print(np.mean(c_gt))
                    # print(np.mod(c_gt))
                    print(len(np.argwhere(c_gt < 5)))
                    print(len(np.argwhere(c_gt < 10)))
                    print(len(np.argwhere(c_gt < 100)))
                    print(len(np.argwhere(c_gt < 200)))
                    print(len(np.argwhere(c_gt < 300)))
                    print(len(np.argwhere(c_gt > 300)))
                    writer.writerow(
                        [file_name, len(c_gt), np.max(c_gt), np.min(c_gt), np.mean(c_gt), len(np.argwhere(c_gt < 5)),
                         len(np.argwhere(c_gt < 10)), len(np.argwhere(c_gt < 100)), len(np.argwhere(c_gt < 200)),
                         len(np.argwhere(c_gt < 250)), len(np.argwhere(c_gt < 300)),
                         len(np.argwhere(c_gt < 350)), len(np.argwhere(c_gt < 400)), len(np.argwhere(c_gt < 500)),
                         len(np.argwhere(c_gt < 1000))])
                    if i > 0: continue
    return

if __name__ == '__main__':
    # gey_all_sks()
    # count_dis()
    # analysis_dis()

    # draw count fig
    plt.figure()
    # ax1 = plt.subplot(121)
    # ax1.title('U-RISC', fontsize=16)
    # ax1.ylabel('y', fontsize=12)

    x = np.array([5,10,50,100,500])
    y = np.array([0.171957846,0.3933278220,0.782581871,0.951584359,1.0])
    std = np.array([0.163421979,0.196660479,0.1525291,0.277346931,0.0])

    plt.errorbar(x, y, yerr=std, fmt='o', color='blue',
                 ecolor='lightblue', elinewidth=3, capsize=0,alpha = 0.5)

    # from scipy.interpolate import make_interp_spline
    #
    # xnew = np.linspace(5, 100, 30)  # 300 represents number of points to make between T.min and T.max
    # y_smooth = make_interp_spline(x, y)( xnew)

    plt.plot(x, y, label='U-RISC')
    plt.plot(x, y, 'or')
    plt.xscale("log")


    y_1 = np.array([0.371957846,0.8933278220,0.982581871,1.0,1.0])
    std_1 = np.array([0.12942,0.1942,0.1525291,0.077346931,0.0])

    plt.plot(x, y_1, label='ISBI 2012')
    plt.errorbar(x, y_1, yerr=std, fmt='*', color='orange',
                 ecolor='bisque', elinewidth=3, capsize=0, alpha = 0.5)
    plt.plot(x, y_1, 'or')
    plt.xscale("log")
    plt.xlabel('Euclidean distance')
    plt.ylabel('Proportion')
    plt.legend()

    plt.show()
    # plt.savefig('dis_count.pdf')

    # # Import Data
    #  # Draw Plot
    # with open('/Users/shiwakaga/Desktop/dis_count/0145_1_1565791505_47.png.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         df = np.array([float(x) for x in row])/5
    # print(df)
    # print(df.shape)
    #
    # plt.figure(figsize=(13, 10), dpi=80)
    #
    # sns.distplot(df.loc[df['class'] == 'compact', "cty"], color="dodgerblue", label="Compact", hist_kws={'alpha': .7},
    #              kde_kws={'linewidth': 3})
    #
    # sns.distplot(df.loc[df['class'] == 'suv', "cty"], color="orange", label="SUV", hist_kws={'alpha': .7},
    #              kde_kws={'linewidth': 3})
    #
    # sns.distplot(df.loc[df['class'] == 'minivan', "cty"], color="g", label="minivan", hist_kws={'alpha': .7},
    #              kde_kws={'linewidth': 3})
    #
    # plt.ylim(0, 0.35)
    # # Decoration
    # plt.title('Density Plot of City Mileage by Vehicle Type', fontsize=22)
    # plt.legend()
    # plt.show()
    # #


