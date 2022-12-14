import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch import nn
import cv2
from skimage import morphology
from .hausedorff_dis import hausdorff_distance_with_t


class PerceptualHausdorfffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

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

    def forward(self, set1, set2, tolerance=2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an M-dimensional point.
        :return: The Perceptual Hausdorfff Distance between set1 and set2.
        """
        set1, set2 = torch.squeeze(set1, 1),torch.squeeze(set2, 1)
        set1, set2 = self.get_points(set1), self.get_points(set2)

        whd = hausdorff_distance_with_t(set1, set2, distance='euclidean', tolerance=tolerance)
        if whd == 0: whd=float(0)
        try:
            res = torch.tensor(whd, requires_grad=True)
        except Exception as e:
            # print(whd)
            return torch.tensor(float(0))

        return res
