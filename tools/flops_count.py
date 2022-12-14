#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: flops_count.py
@time: 2021/11/22 12:26 PM
"""
# import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

from CASENet.modules.CASENet import CASENet_resnet101
from linknet.models.linknet import LinkNet


with torch.cuda.device(0):
    # # U-Net
    # from Image_Segmentation.network import U_Net
    # net = U_Net(img_ch=1, output_ch=1)
    # macs, params = get_model_complexity_info(net, (1,256,256), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # # PHD
    # from Add_PHD_loss.network import U_Net
    # net = U_Net(img_ch=1, output_ch=1)
    # # macs1, params1 = get_model_complexity_info(net, (1,512,512), as_strings=True,
    # #                                          print_per_layer_stat=True, verbose=True)
    # macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # # macs, params = macs1+macs2, params1+params2

    # # CASENet
    # net = CASENet_resnet101()
    # macs, params = get_model_complexity_info(net, (3,512,512), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)

    # # Linknet
    # net = LinkNet()
    # macs, params = get_model_complexity_info(net, (3,512,512), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)

    # GLnet
    from GLNet.models.fpn_global_local_fmreg_ensemble import *
    # resnet_global = resnet50(True) # Computational complexity:       21.52 GMac
                                   #  Number of parameters:           23.51 M
    # resnet_local = resnet50(True)

    # fpn module
    # fpn_global = fpn_module_global(2)
    # fpn_local = fpn_module_local(2)
    # net = fpn(2)
    net = fpn(2)
    macs, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    # # senet
    # import torch.hub
    # hub_model = torch.hub.load(
    #     'moskomule/senet.pytorch',
    #     'se_resnet101',
    #     num_classes=2)
    # macs, params = get_model_complexity_info(hub_model, (3, 512, 512), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)

    # unet++
    # # https://github.com/monchhichizzq/Unet_Medical_Segmentation
    # from UNetplusplus.UNetPP import UNetPlusPlus
    # net = UNetPlusPlus()
    # macs, params = get_model_complexity_info(net, (512,512,3), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
