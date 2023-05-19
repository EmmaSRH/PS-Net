#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: get_slice.py
@time: 2021/8/23 6:11 PM
"""
import os
import shutil
from glob import glob
import csv
import numpy as np
import cv2


def copy2(min_index, max_index, s=1,
          from_dir="L:\\RABBIT",
          to_dir="O:\\"):  # "C:\\Users\\PKU\\Temp"):
    for i in range(min_index, max_index + 1):
        from_img_dir = "%s\\%04d\\8-bit\\%03d" % (from_dir, i, s)
        from_mosaic = "%s\\%04d\\grid.mosaic" % (from_dir, i)
        print(from_img_dir)
        if os.path.exists(from_img_dir):
            os.mkdir("%s\\%04d" % (to_dir, i))
            to_img_dir = "%s\\%04d\\%03d" % (to_dir, i, s)
            to_mosaic = "%s\\%04d\\grid.mosaic" % (to_dir, i)
            shutil.copytree(from_img_dir, to_img_dir)
            shutil.copy(from_mosaic, to_mosaic)


def assemble2cmd(min_index, max_index, s=1, resize=1,
                 data_dir="O:\\",  # "C:\\Users\\PKU\\Temp",
                 to_dir="O:\\",  # "C:\\Users\\PKU\\Temp",
                 ext=".npy", output_cmd="run.cmd"):
    lines = []
    for i in range(min_index, max_index + 1):
        p_dir = "%s\\%04d\\%03d" % (data_dir, i, s)
        if os.path.exists(p_dir):
            mosaic = "%s\\%04d\\grid.mosaic" % (data_dir, i)
            output = "%s\\%04d_%d%s" % (to_dir, i, s * resize, ext)
            # l_fmt = "C:\\Python37\\scripts\\nornir-assemble -i %s -o %s -p %s -s %d\n"
            l_fmt = "O:\\software\\nornir-assemble -i %s -o %s -p %s -s %d\n"
            line = l_fmt % (mosaic, output, p_dir, resize)
            lines.append(line)
    with open("%s\\%s" % (data_dir, output_cmd), "w") as f:
        f.writelines(lines)


def crop_asimg(data_folder="/media/retina/New Volume/ALL", save_folder="/media/retina/New Volume/T",
               img_size=[10000, 10000],
               repeat_n=1, focus_center=True, discard_small=False):
    import pyvips  # install pyvips libvips for read large .png file

    fnames = glob(os.path.join(data_folder, "*.png"))
    assert len(fnames) > 0
    import time
    id = int(time.time())
    fnames = sorted(fnames)

    if img_size is not None:
        assert len(img_size) == 2
        assert img_size[0] % 2 == 0
        assert img_size[1] % 2 == 0
        save_folder = os.path.join(save_folder, "%d_%d" % tuple(img_size))
    else:
        save_folder = os.path.join(save_folder, "full")
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    for fname in fnames:
        print(fname)
        full_img = pyvips.Image.new_from_file(fname, access="sequential")
        data = np.ndarray(buffer=full_img.write_to_memory(), dtype=np.uint8,
                          shape=[full_img.height, full_img.width, full_img.bands])
        cropeds = []
        if img_size is not None:
            if data.shape[0] == img_size[0] and data.shape[1] == img_size[1]:
                cropeds = [data]
            else:
                while True:
                    if focus_center:
                        center = [data.shape[0] / 2, data.shape[1] / 2]
                    else:
                        x_center = int(np.random.random() * data.shape[0])
                        y_center = int(np.random.random() * data.shape[1])
                        center = [x_center, y_center]
                    x_from = int(center[0] - img_size[0] / 2)
                    x_to = int(center[0] + img_size[0] / 2) + 1
                    y_from = int(center[1] - img_size[1] / 2)
                    y_to = int(center[1] + img_size[1] / 2)
                    img = data[x_from:x_to, y_from:y_to]
                    if img.shape[0] < img_size[0] or img.shape[1] < img_size[1]:
                        if discard_small:
                            continue
                    cropeds.append(img)
                    if len(cropeds) == repeat_n:
                        break
        else:
            cropeds = [data]
        # img = (data*255).astype(np.uint8)
        name = os.path.split(fname)[-1]
        for index, img in enumerate(cropeds):
            s_fname = os.path.join(save_folder, name).replace(".png", "_%d_%d.png" % (id, index))
            print("save %s image into %s" % (str(img.shape), s_fname))
            # return s_fname, img, data
            cv2.imwrite(s_fname, img)
        del full_img
        del data


def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        lines = csv.reader(csvfile)
        infos = []
        for row in lines:
            # print(row)
            if row[0] != 'node_id':
                infos.append([row[0],float(row[1]),float(row[2]), int(row[3]), float(row[4])])
            # node_id, x, y, z, size
    return infos


def change_fname(info):
    """
    change int to fname
    :param fname:
    :return:
    """
    if info < 10:
        return '000' + str(info)
    if info >= 10 and info < 100:
        return '00' + str(info)
    if info > 100:
        return '0' + str(info)


def copy_part_image(infos, img_path, save_path):
    """
    use cell infos to crop img and save new img
    :param infos: (node_id, x, y, z, size) str,float,float,int,float
    :param img_path: path to read img, str
    :param save_path: path to save img, str
    :return: img
    """
    print(infos)
    fname = img_path + change_fname(infos[3]) + '_1.png'
    print(fname)
    # full_img = cv2.imread(fname)
    # print(full_img.shape)
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    import pyvips  # install pyvips libvips for read large .png file
    full_img = pyvips.Image.new_from_file(fname, access="sequential")
    data = np.ndarray(buffer=full_img.write_to_memory(), dtype=np.uint8,
                      shape=[full_img.height, full_img.width, full_img.bands])
    #
    # print(data.size)

    x_from = int(infos[1] - infos[-1])
    x_to = int(infos[1] + infos[-1])
    y_from = int(infos[2] - infos[-1])
    y_to = int(infos[2] + infos[-1])
    img = data[x_from:x_to, y_from:y_to]

    print(x_from, x_to, y_from, y_to)

    # exit()

    s_fname = save_path + infos[0] + '_' + change_fname(infos[3]) + '.png'
    print("save image into %s" % (s_fname))
    cv2.imwrite(s_fname, img)

    del full_img
    del data


if __name__ == '__main__':
    cell_files = glob('/home/srh/Add_PHD_loss/tlps/data/*.csv')
    for cell_file in cell_files:
        print(cell_file)
        cell_name = cell_file.split('/')[-1].split('_')[-1].split('.')[0]
        infos = read_csv(cell_file) #
        img_path = '/mnt/srh/RC1_Raw_DATA/full_img/'
        save_path = '/mnt/srh/RC1_Raw_DATA/crop_cells/' + cell_name + '/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for i in range(len(infos)):
            copy_part_image(infos[0], img_path, save_path)
