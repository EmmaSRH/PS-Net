#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: generate_tiff.py
@time: 2021/8/20 2:12 PM
"""
import cv2

from libtiff import TIFF
# to open a tiff file for reading:

tif = TIFF.open('filename.tif', mode='r')
# to read an image in the currect TIFF directory and return it as numpy array:

image = tif.read_image()
# to read all images in a TIFF file:

for image in tif.iter_images(): # do stuff with image
# to open a tiff file for writing:

tif = TIFF.open('filename.tif', mode='w')
# to write a image to tiff file

tif.write_image(image)