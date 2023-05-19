from skimage.util import view_as_windows as viewW
import numpy as np
from scipy.ndimage.filters import convolve
import cv2
import torch
import torch.nn.functional as F
import pdb
from torch.autograd import gradcheck

def view_as_windows_torch(image, side_size, stride=None):
    """
    image: 2D tensor
    """
    if stride is None:
        stride = side_size[0] // 2, side_size[1] // 2

    windows = image.unfold(0, side_size[0], stride[0])
    return windows.unfold(1, side_size[1], stride[1])

def loop(arr):
    side_size = (3, 3)
    ext_size = (side_size[0] - 1) // 2, (side_size[1] - 1) // 2
    img = F.pad(arr, (ext_size[0], ext_size[1],ext_size[0], ext_size[1]), 'constant', value=0)
    out = view_as_windows_torch(img, side_size)
    out = out.reshape(out.shape[0:2] + (9,))
    out = out[:, :, torch.tensor([1, 2, 5, 8, 7, 6, 3, 0, 1])]
    out[:, :, -1] = out[:, :, 0]
    n_0to1 = torch.zeros(arr.shape)
    n_0to1[:, :] = torch.sum(torch.diff(out[:, :], axis=2) == 1, axis=2)
    n_0to1[0, :] = 0
    n_0to1[-1, :] = 0
    n_0to1[:, 0] = 0
    n_0to1[:, -1] = 0
    return n_0to1
def skeletonize(input, term_thres=0):
    """
    input: 1xWxH Tensor
    return: 1xWxH Tensor
    """
    n_step1, n_step2 = torch.inf, torch.inf
    thining_mask = F.relu(input.unsqueeze(0))
    thining_mask = torch.where(thining_mask>0, torch.ones_like(thining_mask), thining_mask)

    while n_step1 > term_thres or n_step2 > term_thres:
        step10 = torch.where(thining_mask >= 1, torch.ones_like(thining_mask), torch.zeros_like(thining_mask))>0
        filter = torch.ones((3, 3)).type(torch.float32).unsqueeze(0).unsqueeze(0)
        step11_mask = F.conv2d(input,filter, padding=1)
        step11 = torch.clamp(step11_mask, min=2, max=6) > 0

        step12 = loop(thining_mask[0,0,:,:].clone()) == 1

        step13_kernel = torch.tensor([[0, -1, 0], [0, 0, -1], [0, -1, 0]]).type(torch.float32).unsqueeze(0).unsqueeze(0)
        step13_mask =  F.conv2d(thining_mask, step13_kernel, padding=1)
        step13 = torch.clamp(step13_mask, min=-3) > -3

        step14_kernel = torch.tensor([[0, 0, 0], [-1, 0, -1], [0, -1, 0]]).type(torch.float32).unsqueeze(0).unsqueeze(0)
        step14_mask =  F.conv2d(thining_mask, step14_kernel, padding=1)
        step14 = torch.clamp(step14_mask, min=-3) > -3
        step1_mask = step10 & step11 & step12 & step13 & step14

        n_step1 = torch.count_nonzero(step1_mask)
        thining_mask = torch.where(step1_mask, torch.zeros_like(thining_mask), thining_mask)
        step20 = torch.where(thining_mask >= 1, torch.ones_like(thining_mask), torch.zeros_like(thining_mask))>0

        step21_kernel = torch.ones((3, 3)).type(torch.float32).unsqueeze(0).unsqueeze(0)
        step21_mask = F.conv2d(thining_mask, step21_kernel, padding=1)
        step21 = torch.clamp(step21_mask, min=2, max=6) > 0

        step22 = loop(thining_mask[0,0,:,:].clone()) == 1

        step23_kernel = torch.tensor([[0, -1, 0], [-1, 0, -1], [0, 0, 0]]).type(torch.float32).unsqueeze(0).unsqueeze(0)
        step23_mask = F.conv2d(thining_mask, step23_kernel, padding=1)
        step23 = torch.clamp(step23_mask, min=-3) > -3

        step24_kernel = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, -1, 0]]).type(torch.float32).unsqueeze(0).unsqueeze(0)
        step24_mask = F.conv2d(thining_mask, step24_kernel, padding=1)
        step24 = torch.clamp(step24_mask, min=-3) > -3
        step2_mask = step20 & step21 & step22 & step23 & step24

        n_step2 = torch.nonzero(step2_mask, as_tuple=False).size(0)
        thining_mask = torch.where(step2_mask, torch.zeros_like(thining_mask), thining_mask)

    return thining_mask


if __name__ == '__main__':
    binary_img = 1-cv2.imread('test.png',0)/255
    input_tensor = torch.tensor(binary_img).unsqueeze(0).type(torch.float32)
    print(input_tensor.shape)
    import time
    start = time.time()
    skeleton = skeletonize(input_tensor)
    use_time = time.time() - start
    print('use time: ', use_time)
    out = skeleton.detach().numpy()
    cv2.imwrite('skel-torch.png',out[0,0,:,:]*255)