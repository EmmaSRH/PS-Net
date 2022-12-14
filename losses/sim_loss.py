import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch import nn
import cv2
from skimage import morphology
from .hausedorff_dis import hausdorff_distance_with_t
import torch.nn.functional as F


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def get_points(self,arr):
        for b in range(arr.shape[0]):
            # print(arr[b,:].shape) # torch.Size([512, 512])
            arr = arr.cpu().detach().numpy()
            _, binary = cv2.threshold(arr[b, :], 0.5, 1, cv2.THRESH_BINARY_INV)
            binary[binary == 1] = 1
            skeleton = morphology.skeletonize(binary)
            # print(skeleton)

            # use cpu hausdorff
            points = np.argwhere(skeleton == True)
            # print(points)

            # # use pairwise dis
            # points = (skeleton > 0.5).nonzero()
            # points = torch.tensor(points)

            return points
    def get_local_pre(self,pre_locals):
        N,c,h,w = pre_locals[0].shape[0],pre_locals[0].shape[1],pre_locals[0].shape[2],pre_locals[0].shape[3]
        pre_local = torch.zeros((N,c,h*2,w*2))
        pre_local[:,:,:h,:w] = pre_locals[0]
        pre_local[:,:,:h,w:] = pre_locals[1]
        pre_local[:,:,h:,:w] = pre_locals[2]
        pre_local[:,:,h:,w:] = pre_locals[3]
        # print(pre_local.shape)
        pre_local = F.interpolate(pre_local, size=(512,512))
        # print('pre_local.shape', pre_local.shape)
        # exit()
        return pre_local

    def forward(self, pre_global, pre_locals, tolerance=2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1:  (N,1,512,512)
        :param set2:  [(N,1,512,512)*4]
        :return: The Perceptual Hausdorfff Distance between set1 and set2.
        """
        pre_local =self.get_local_pre(pre_locals)
        set1, set2 = torch.squeeze(pre_global, 1), torch.squeeze(pre_local, 1)

        set1, set2 = self.get_points(set1), self.get_points(set2)

        whd = hausdorff_distance_with_t(set1, set2, distance='euclidean', tolerance=tolerance)
        if whd == 0: whd=float(0)
        try:
            res = torch.tensor(whd, requires_grad=True)
        except Exception as e:
            # print(whd)
            return torch.tensor(float(0))

        return res
