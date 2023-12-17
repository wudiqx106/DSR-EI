import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
import random
from network_utils import gradient
from utils import add_noise
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F


def img_read(filename):
    img_file = Image.open(filename)
    img = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return img


def depth_read(filename):
    img_file = Image.open(filename)
    depth = np.array(img_file, dtype=float)  # in the range [0,255]
    img_file.close()
    return depth


# dataloader by Xin Qiao 2022.4.8
class RGBDD(Dataset):
    """
    RGBB-D-D Dataset.
    max range expect for lights: 5m
    """
    def __init__(self, root='/home/qiaoxin/prj/Qiao/dataset/RGBDD', scale=4, split='train', augment=True,
                 downsample='bicubic', pre_upsample=False, to_pixel=False, input_size=None, noisy=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale

        """
        self.input_size = input_size
        self.pre_upsample = pre_upsample
        self.augment = augment
        self.split = split
        self.downsample = downsample
        self.noisy = noisy
        self.root_dir = root
        self.scale = scale
        height, width = (input_size, input_size)
        crop_size = (input_size, input_size)
        self.height = height
        self.width = width
        self.height = height
        self.width = width
        self.crop_size = crop_size

        if self.split == 'train':
            self.data_path = os.path.join(self.root_dir, 'train')
            self.list_dir = os.listdir(self.data_path)
        else:
            self.data_path = os.path.join(self.root_dir, 'test')
            self.list_dir = sorted(os.listdir(self.data_path))

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.data_path, self.list_dir[idx], self.list_dir[idx]+'_HR_gt.png')
        img_path = os.path.join(self.data_path, self.list_dir[idx], self.list_dir[idx]+'_RGB.jpg')
        depth_hr = depth_read(gt_path)
        image = img_read(img_path)

        if self.input_size is not None:
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0+self.input_size, y0:y0+self.input_size]
            depth_hr = depth_hr[x0:x0+self.input_size, y0:y0+self.input_size]

        h, w = image.shape[:2]
        s = self.scale
        if self.downsample == 'bicubic':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//s, h//s), Image.BICUBIC).resize((w, h), Image.BICUBIC))
        else:
            raise NotImplementedError

        if self.noisy:
            depth_lr = add_noise(depth_lr, sigma=0.04, inv=False)

        # gradient
        GT_grad = gradient(depth_hr)

        # normalize
        depth_min = 0
        depth_max = 5000
        assert depth_min != depth_max
        depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)
        depth_lr = (depth_lr - depth_min) / (depth_max - depth_min)
        depth_grad = GT_grad / (depth_max - depth_min)
        image = image.astype(np.float32).transpose(2, 0, 1) / 255

        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        # follow DKN, use bicubic upsampling of PIL
        depth_lr_up = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))

        if self.pre_upsample:
            depth_lr = depth_lr_up

        # to tensor
        image = torch.from_numpy(image).float()
        depth_hr = torch.from_numpy(depth_hr).unsqueeze(0).float()
        depth_grad = torch.from_numpy(depth_grad).unsqueeze(0).float()
        depth_lr = torch.from_numpy(depth_lr).unsqueeze(0).float()

        # transform
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            image = augment(image)
            depth_hr = augment(depth_hr)
            depth_lr = augment(depth_lr)
            # depth_lr_up = augment(depth_lr_up)

        image = image.contiguous()
        depth_hr = depth_hr.contiguous()
        depth_lr = depth_lr.contiguous()

        data = {'image': image,
                'lr': depth_lr,
                'hr': depth_hr,
                'grad': depth_grad,
                'min': 0,
                'max': depth_max / 10}

        return data