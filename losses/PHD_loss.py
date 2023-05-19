import torch
from torch import nn
import torch.nn.functional as F
from .skeleton_pytorch  import *

class PerceptualHausdorfffLoss(nn.Module):
    # def __init__(self):
    #     super(nn.Module, self).__init__()
    def __init__(self) -> None:
        super(PerceptualHausdorfffLoss, self).__init__()

    def get_points(self,arr):
        """
        Get the skeleton points.
        :param arr: 1xWxH Tensor
        :res: a list of the NxWxH points,
        """
        skeleton_mask = skeletonize(arr).squeeze(0)
        coords = torch.nonzero(skeleton_mask.squeeze(0))
        skeleton_masks = F.relu(skeleton_mask.repeat(len(coords), 1, 1)-1)
        for i in range(len(coords)):
            skeleton_masks[i,coords[i][0],coords[i][1]]+=1.0
        return skeleton_masks.view(skeleton_masks.shape[0],skeleton_masks.shape[1]*skeleton_masks.shape[2])

    def cdist(self,x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
              i.e. dist[i,j] = ||x[i,:]-y[j,:]||

        """
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences ** 2, -1).sqrt()
        return distances
    def p_hausdorff(self,XA, XB, tolerance=1):
        """
        Compute the perceptual hausdorff distance
        :param XA: NxWxH Tensor
        :param XB: MxWxH Tensor
        """
        d1_matrix = torch.cdist(XA, XB)
        d2_matrix = torch.cdist(XB, XA)
        res = torch.mean(torch.min(F.relu(d1_matrix - tolerance),1)[0]) + torch.mean(
            torch.min(F.relu(d2_matrix - tolerance),1)[0])
        return res

    def forward(self, pre, gt, tolerance=1):
        """
        Compute the Perceptual Hausdorff Distance function.
        :param pre: Bx1xWxH Tensor
        :param gt: Bx1xWxH Tensor
        :return: The Perceptual Hausdorfff Distance between pre and gt.
        """
        phd = torch.zeros(pre.shape[0])
        for b in range(pre.shape[0]):
            points1, points2 = self.get_points(pre[b,::]), self.get_points(gt[b,::])
            phd[b] = self.p_hausdorff(points1, points2, tolerance=tolerance)
        return torch.mean(phd)


if __name__ == '__main__':
    import cv2
    inputs1 = torch.tensor(cv2.imread('test.png',0)/255,requires_grad=True).unsqueeze(0).unsqueeze(0).float()
    inputs2 = torch.tensor(cv2.imread('test1.png',0)/255).unsqueeze(0).unsqueeze(0).float()
    loss = PerceptualHausdorfffLoss()
    score = loss(inputs1,inputs2)
    print(score)

