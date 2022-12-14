#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: read_tlq.py
@time: 2021/8/23 6:20 PM
"""
import glob
import csv

def read_tlq(file_path):
    with open(file_path,encoding='utf-8') as f:
        line_num = 0
        locations = []
        for line in f:
            # print('Start to read line {}. '.format(line_num))
            line_num += 1
            strs = line.split(' ')
            if len(strs)==5 and len(strs[2]) == 19 and strs[3][:2] == 'Y:':
                # print(line)
                node_id = int(strs[1])
                x = float(strs[2].split(':')[1])
                y = float(strs[3].split(':')[1])
                z = int(strs[4].split(':')[1][:-3])
                # print(node_id,x,y,z)
                locations.append([node_id,x,y,z])
        return locations
def get_size(file_path, node_num):
    with open(file_path,encoding='utf-8') as f:
        lines = f.readlines()
        # print(len(lines))
        size = []
        start_line = 0
        for i in range(len(lines)):
            # print('Start to read line {}. '.format(line_num))
            strs = lines[i].split(' ')
            # print(strs)
            if len(strs)==4 and strs[2] == 'size':
                print(i)
                start_line = i
        for i in range(start_line+2,start_line+node_num+2):
            print(lines[i])
            size_tmp = float(lines[i].split(' ')[2].split(',')[0][2:])
            print(size_tmp)
            size.append(size_tmp)
        return size

if __name__ == '__main__':
    tlq_files = glob.glob('/Users/shiwakaga/Desktop/tlps/*.tlp')
    save_path = '/Users/shiwakaga/Desktop/tlps/data/'
    for tlq_file in tlq_files:
        tlq_id = tlq_file.split('/')[-1].split(' ')[0].split('-')[1]
        print('Start to read number {} cell'.format(tlq_id))
        locations = read_tlq(tlq_file)
        size = get_size(tlq_file, len(locations))
        with open(save_path+'file_{}.csv'.format(tlq_id), 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["node_id", "x", "y", "z","size"])
            # 写入多行用writerows
            for i in range(len(locations)):
                writer.writerow(locations[i]+[size[i]])
