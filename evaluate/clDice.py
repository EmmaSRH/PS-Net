from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import cv2
def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

if __name__ == '__main__':
    # img1 = cv2.imread('../losses/test.png', 0)/255
    # img2 = cv2.imread('../losses/test.png', 0)/255

    img1 = 1 - cv2.imread('/Users/semma/Desktop/0078_1_1565791505_0/0_2048_0_2048.png', 0) / 255
    img2 = 1 - cv2.resize(cv2.imread('/Users/semma/Desktop/test/0078_1_1565791505_0/0_2048_0_2048.jpg', 0),
                          (512, 512)) / 255
    score = clDice(img1, img2)
    print(score)

