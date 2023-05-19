import torch
from torch import nn
import torch.nn.functional as F
from .skeleton_pytorch  import *

class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()
    def get_points(self, arr):
        """
        Get the skeleton points.
        :param arr: 1xWxH Tensor
        :res: a list of the NxWxH points,
        """
        skeleton_mask = skeletonize(arr).squeeze(0)
        coords = torch.nonzero(skeleton_mask.squeeze(0))
        skeleton_masks = F.relu(skeleton_mask.repeat(len(coords), 1, 1) - 1)
        for i in range(len(coords)):
            skeleton_masks[i, coords[i][0], coords[i][1]] += 1.0
        return skeleton_masks.view(skeleton_masks.shape[0], skeleton_masks.shape[1] * skeleton_masks.shape[2])
    def get_local_pre(self,pre_locals):
        N,c,h,w = pre_locals[0].shape[0],pre_locals[0].shape[1],pre_locals[0].shape[2],pre_locals[0].shape[3]
        pre_local = torch.zeros((N,c,h*2,w*2))
        pre_local[:,:,:h,:w] = pre_locals[0]
        pre_local[:,:,:h,w:] = pre_locals[1]
        pre_local[:,:,h:,:w] = pre_locals[2]
        pre_local[:,:,h:,w:] = pre_locals[3]
        pre_local = F.interpolate(pre_local, size=(512,512))

        return pre_local
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
    def forward(self, pre_global, pre_locals, tolerance=2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1:  (N,1,W,H)
        :param set2:  [(N,1,W,H)*4]
        :return: The Perceptual Hausdorfff Distance between set1 and set2.
        """
        sim = torch.zeros(pre_global.shape[0])
        pre_local =self.get_local_pre(pre_locals)
        for b in range(pre_local.shape[0]):
            set1, set2 = self.get_points(pre_global), self.get_points(pre_local)
            sim[b] = self.p_hausdorff(set1, set2, tolerance=tolerance)
        return torch.mean(sim)

