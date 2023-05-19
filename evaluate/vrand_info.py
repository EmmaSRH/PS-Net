#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: vrand_info.py
@time: 2021/10/12 6:50 PM
"""
import os

# allow scripts running
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
import imagej
import re
warnings.filterwarnings("ignore")

# use ij to evaluate the performance by Vrand and Vinfo. These are the scripts
ij = imagej.init('/home/srh/Fiji.app')
# ij = imagej.init('/Applications/Fiji.app')

Language_extension = "BeanShell"

macroVRand = """
import trainableSegmentation.metrics.*;
#@output String VRand
import ij.IJ;
originalLabels=IJ.openImage("AAAAA");
proposedLabels=IJ.openImage("BBBBB");
metric = new RandError( originalLabels, proposedLabels );
maxThres = 1.0;
maxScore = metric.getMaximalVRandAfterThinning( 0.0, maxThres, 0.1, true );  
VRand = maxScore;
"""

macroVInfo = """
import trainableSegmentation.metrics.*;
#@output String VInfo
import ij.IJ;
originalLabels=IJ.openImage("AAAAA");
proposedLabels=IJ.openImage("BBBBB");
metric = new VariationOfInformation( originalLabels, proposedLabels );
maxThres =1.0;
maxScore = metric.getMaximalVInfoAfterThinning( 0.0, maxThres, 0.1 );  
VInfo = maxScore;
"""


# evaluate function, input: image path and label path, output: two scores
def evl(image_path, label_path):
    reg1 = re.compile('AAAAA')
    macror = reg1.sub(label_path, macroVRand)
    macroi = reg1.sub(label_path, macroVInfo)

    reg2 = re.compile('BBBBB')
    macror = reg2.sub(image_path, macror)
    macroi = reg2.sub(image_path, macroi)

    VRand = float(
        str(ij.py.run_script(Language_extension, macror).getOutput('VRand')))
    VInfo = float(
        str(ij.py.run_script(Language_extension, macroi).getOutput('VInfo')))

    return VRand, VInfo

if __name__ == '__main__':
    pre_path = '/home/srh/Add_PHD_loss/predictions/prediction-2048-all/U_Net/U_Net-399_pre_ful/0078_1_1565791505_0.png'
    gt_path = '/mnt/srh/U-RISC-DATASET/label/complex/test/0078_1_1565791505_0.png'

    # pre_path = '/Users/shiwakaga/Desktop/predictions/prediction-2048-focal+dice+bce-7-22/pre_ful/0078_1_1565791505_0.png'
    # gt_path = '/Users/shiwakaga/Downloads/U-RISC-DATASET/label/complex/test/0078_1_1565791505_0.png'

    vrand, vinfo = evl(pre_path, gt_path)
    print(vrand, vinfo)