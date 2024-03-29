import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import glob
import cv2


class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=2048,resize_size = 512,mode='train',augmentation_prob=0.4,type='urisc'):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        # self.GT_paths = root
        # self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

        if type == "urisc":
            self.image_paths = glob.glob(root + '*/*.png')

        if type == "isbi":
            self.image_paths = glob.glob(root + '*.png')

        if type == "road":
            self.image_paths = glob.glob(root + '*.tiff')

        if type == "cracktree":
            self.image_paths = glob.glob(root + '*.jpg')

        if type == "drive":
            self.image_paths = glob.glob(root + '*.tif')

        # # 2012
        # self.image_paths = glob.glob(root + '*.png')

        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        self.resize_size = resize_size
        self.type = type
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def get_local_imgs(self, img):
        crop_imgs = torch.zeros((16, img.shape[0], self.resize_size, self.resize_size))
        cnt = 0
        for i in range(4):
            for j in range(4):
                x_1, x_2 = self.resize_size * i, self.resize_size * i + self.resize_size
                y_1, y_2 = self.resize_size * j, self.resize_size * j + self.resize_size
                crop_imgs[cnt, :, :, :] = img[:, x_1:x_2, y_1:y_2]
                cnt += 1
        return crop_imgs

    def get_local_imgs_4_patch(self, img):
        crop_imgs = torch.zeros((4, img.shape[0], self.resize_size, self.resize_size))
        cnt = 0
        for i in range(2):
            for j in range(2):
                x_1, x_2 = self.resize_size * 2 * i, self.resize_size * 2 * i + self.resize_size * 2
                y_1, y_2 = self.resize_size * 2 * j, self.resize_size * 2 * j + self.resize_size * 2
                Resize_ = T.Resize((512, 512))
                crop_imgs[cnt, :, :, :] = Resize_(img[:, x_1:x_2, y_1:y_2])
                cnt += 1
        return crop_imgs
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        image_path = self.image_paths[index]

        if self.type == "urisc":
            filename = image_path.split('/')[-2]+'/'+image_path.split('/')[-1]
            GT_path = image_path.replace('.png','.jpg')
            gt_pt_path = image_path.replace('test','test_sk')

        if self.type == "isbi":
            filename = image_path.split('/')[-1]
            GT_path = image_path.replace('imgs','labels')

        if self.type == "road":
            filename = image_path.split('/')[-1]
            GT_path = image_path.replace(image_path.split('/')[-2],image_path.split('/')[-2]+'_labels').replace('.tiff','.tif')

        if self.type == "cracktree":
            filename = image_path.split('/')[-1]
            GT_path = image_path.replace('image','gt').replace('.jpg','.bmp')

        if self.type == "drive":
            filename = image_path.split('/')[-1]
            GT_path = image_path.replace('images','mask').replace('.tif','_mask.gif')

            image = Image.open(image_path).convert('RGB')
            GT = Image.open(GT_path).convert('L')

            aspect_ratio = image.size[1] / image.size[0]
            Transform = []

            # add resize
            # ResizeRange = random.randint(300,320)
            # ResizeRange = random.randint(self.resize_size,self.resize_size)
            # Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))

            p_transform = random.random()

            if (self.mode == 'train') and p_transform <= self.augmentation_prob:
                RotationDegree = random.randint(0, 3)
                RotationDegree = self.RotationDegree[RotationDegree]
                if (RotationDegree == 90) or (RotationDegree == 270):
                    aspect_ratio = 1 / aspect_ratio

                Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

                RotationRange = random.randint(-10, 10)
                Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                # CropRange = random.randint(250,270)
                # CropRange = random.randint(self.resize_size, self.resize_size)
                # Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                Transform = T.Compose(Transform)

                image = Transform(image)
                GT = Transform(GT)

                # ShiftRange_left = random.randint(0, 20)
                # ShiftRange_upper = random.randint(0, 20)
                # ShiftRange_right = image.size[0] - random.randint(0, 20)
                # ShiftRange_lower = image.size[1] - random.randint(0, 20)
                # image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
                # GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

                if random.random() < 0.5:
                    image = F.hflip(image)
                    GT = F.hflip(GT)

                if random.random() < 0.5:
                    image = F.vflip(image)
                    GT = F.vflip(GT)

                Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

                image = Transform(image)

                Transform = []

            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)

            image = Transform(image)  # (2048,2048)
            GT = Transform(GT)  # (2048,2048)

            img_crop_list = self.get_local_imgs_4_patch(image)
            gt_crop_list = self.get_local_imgs_4_patch(GT)

            Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            Resize_ = T.Resize((512, 512))
            image = Resize_(Norm_(image))
            GT = Resize_(GT)

            return filename, image, GT, img_crop_list, gt_crop_list

        def __len__(self):
            """Returns the total number of font files."""
            return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, resize_size=512, mode='train',augmentation_prob=0.4,type="urisc"):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root = image_path, image_size =image_size, resize_size = resize_size, mode=mode,augmentation_prob=augmentation_prob,type=type)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
