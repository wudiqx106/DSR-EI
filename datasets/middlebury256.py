# no way. performance is poor

import random
from pathlib import Path
from network_utils import gradient
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
from .utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip, \
    read_calibration, create_depth_from_pfm
from utils import add_noise



VAL_SET_2005_2006 = ['Moebius', 'Lampshade1', 'Lampshade2']
VAL_SET_2014 = ['Shelves-perfect', 'Playtable-perfect']
TEST_SET_2005_2006 = ['Reindeer', 'Bowling1', 'Bowling2']
TEST_SET_2014 = ['Adirondack-perfect', 'Motorcycle-perfect']


class MiddleburyDataset256(Dataset):

    def __init__(
            self,
            root='/home/ubuntu/new_disk/qiaoxin/data/Middlebury/',
            datasets=('2005', '2006', '2014'),
            split='train',
            crop_size=(256, 256),
            do_horizontal_flip=True,
            max_rotation_angle=0,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            use_ambient_images=False,
            crop_deterministic=True,
            scaling=8,
            noisy=False,
            **kwargs
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)

        # if max_rotation_angle > 0 and crop_deterministic:
        #     max_rotation_angle = 0
        #     # print('Set max_rotation_angle to zero because of deterministic cropping')

        self.split = split
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.data = []
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling
        self.noisy = noisy

        root = Path(root)

        # read in various Middlebury datasets using the respective global load_{name} function
        for name in ('2005', '2006', '2014'):
            if name in datasets:
                self.data.extend(globals()[f'load_{name}'](root / name, 1.0, scale_interpolation,
                                                           use_ambient_images, split))
        if self.crop_deterministic:
            assert not use_ambient_images
            # construct deterministic mapping
            self.deterministic_map = []
            for i, datum in enumerate(self.data):
                H, W = datum[0][0].shape[1:]
                num_crops_h, num_crops_w = H // self.crop_size[0], W // self.crop_size[1]
                self.deterministic_map.extend(((i, j, k) for j in range(num_crops_h) for k in range(num_crops_w)))

    def __getitem__(self, index, called=0):
        if called >= 32:  # it has been called 32 times, enough of that
            raise ValueError

        if self.crop_deterministic:
            im_index, crop_index_h, crop_index_w = self.deterministic_map[index]
        else:
            im_index = index

        image, depth_map = random.choice(self.data[im_index])
        # print(image.dtype)
        # print(image.shape)

        # image, depth_map = np.array(image.clone()*255), np.array(depth_map.clone().squeeze(), dtype=np.float32).T
        image, depth_map = image.clone(), depth_map.clone()
        # print('image:', image)
        # print('depth_map:', depth_map)

        # if self.do_horizontal_flip and not self.crop_deterministic:
        #     image, depth_map = random_horizontal_flip((image, depth_map))
        #
        # if self.max_rotation_angle > 0 and not self.crop_deterministic:
        #     image, depth_map = random_rotate((image, depth_map), self.max_rotation_angle, self.rotation_interpolation)
        #     # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
        #     depth_map[depth_map == 0.] = np.nan

        if self.crop_deterministic:
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            image, depth_map = image[:, slice_h, slice_w], depth_map[:, slice_h, slice_w]
        else:
            image, depth_map = random_crop((image, depth_map), self.crop_size)

        image = np.array(image*255).T.astype(np.uint8)
        depth_map = np.array(depth_map.squeeze(), dtype=np.float32).T

        # print('image:', image.shape)

        # plt.imshow(depth_map[:,:])
        # plt.title('full')
        # plt.show()

        # if self.image_transform is not None:
        #     image = self.image_transform(image)
        # if self.depth_transform is not None:
        #     depth_map = self.depth_transform(depth_map)

        h, w = image.shape[:2]
        source = np.array(Image.fromarray(depth_map).resize((w//self.scaling, h//self.scaling), Image.BICUBIC)) # bicubic, RMSE=7.13
        if self.noisy:
            source = add_noise(source, sigma=651)

        # 梯度图
        depth_grad = gradient(depth_map)

        # normalize
        depth_min = np.nanmin(depth_map)
        depth_max = np.nanmax(depth_map)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)   # torch.Size([1, 256, 256])
        source = (source - depth_min) / (depth_max - depth_min)
        depth_grad = depth_grad / (depth_max - depth_min)

        # print('depth_map', depth_map.shape)

        image = image.astype(np.float32).transpose(2, 0, 1) / 255
        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        y_bicubic = np.array(Image.fromarray(source).resize((w, h), Image.BICUBIC))

        source = torch.from_numpy(source).unsqueeze(0).float()
        y_bicubic = torch.from_numpy(y_bicubic).unsqueeze(0).float()
        image = torch.from_numpy(image).float()
        depth_map = torch.from_numpy(depth_map).unsqueeze(0).float()
        depth_grad = torch.from_numpy(depth_grad).unsqueeze(0).float()

        mask_hr = (~torch.isnan(depth_map)).float()
        mask_bi = (~torch.isnan(y_bicubic)).float()
        mask_lr = (~torch.isnan(source)).float()
        mask_grad = (~torch.isnan(depth_grad)).float()
        mask_hr = (mask_hr*mask_grad*mask_bi)    # torch.Size([1, 256, 256])

        depth_map[mask_hr == 0.] = 0.
        depth_grad[mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.
        y_bicubic[mask_hr == 0.] = 0.

        # plt.imshow(depth_map.squeeze().detach().numpy())
        # plt.title('gt')
        # plt.show()
        # plt.imshow(y_bicubic.squeeze().detach().numpy())
        # plt.title('bicubic')
        # plt.show()
        # plt.imshow(image.squeeze().detach().permute(1,2,0).numpy())
        # plt.title('image')
        # plt.show()
        # print('')

        # y_bicubic = np.array(Image.fromarray(source.squeeze().numpy()).resize((w, h), Image.BICUBIC))
        # y_bicubic = torch.from_numpy(y_bicubic).unsqueeze(0).float()

        # y_bicubic = torch.from_numpy(bicubic_with_mask(
        #     source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
        # y_bicubic = y_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))
        # mask_hr = mask_hr.unsqueeze(0)

        # if self.split == 'train' and (torch.mean(mask_lr) < 0.9 or torch.mean(mask_hr) < 0.8):
        #     # omit patch due to too many depth holes, try another one
        #     return self.__getitem__(index, called=called + 1)
        # print('image:', image.shape, 'lr:', y_bicubic.shape, 'depth_map:', depth_map.shape, 'grad:', depth_grad.shape)

        try:
            return {'image': image, 'hr': depth_map, 'mask_hr': mask_hr, 'mask_lr': mask_lr, 'idx': index,
                    'lr': y_bicubic, 'grad': depth_grad, 'max': depth_max/10, 'min': depth_min/10}
        except:
            return self.__getitem__(index, called=called + 1)

    def __len__(self):
        return len(self.deterministic_map if self.crop_deterministic else self.data)


def load_2014(root: Path, scale, scale_interpolation, use_ambient_images, split):
    data = []
    for scene in sorted(root.iterdir()):
        # ignore scenes with imperfect rectification, these are only included in the 2014 dataset anyway
        if not scene.is_dir() or scene.name.endswith('-imperfect'):
            continue

        # make train val test split
        last_dir = scene.parts[-1]

        if (split == 'test' and last_dir in TEST_SET_2014) or (split == 'val' and last_dir in VAL_SET_2014) or \
                (split == 'train' and (last_dir not in TEST_SET_2014) and (last_dir not in VAL_SET_2014)):
            calibration = read_calibration(scene / 'calib.txt')

            # add left and right view, as well as corresponding depth maps
            for view in (0, 1):
                resize = Resize((int(int(calibration['height']) * scale), int(int(calibration['width']) * scale)),
                                scale_interpolation)
                depth_map = resize(torch.from_numpy(create_depth_from_pfm(scene / f'disp{view}.pfm', calibration)))
                transform = Compose([ToTensor(), resize])
                if use_ambient_images:
                    data.append(
                        [(transform(Image.open(path)), depth_map) for path in scene.glob(f'ambient/L*/im{view}*.png')])
                else:
                    data.append([(transform(Image.open(scene / f'im{view}.png')), depth_map)])

    return data


def load_2006(root: Path, scale, scale_interpolation, use_ambient_images, split):
    f = 3740  # px
    baseline = 160  # mm

    data = []
    for scene in sorted(root.iterdir()):
        if not scene.is_dir():
            continue

        # make train val test split
        last_dir = scene.parts[-1]
        if (split == 'test' and last_dir in TEST_SET_2005_2006) or (
                split == 'val' and last_dir in VAL_SET_2005_2006) or (
                split == 'train' and (last_dir not in TEST_SET_2005_2006) and (last_dir not in VAL_SET_2005_2006)):

            # add left and right view, as well as corresponding depth maps
            for view in (1, 5):
                disparities = torch.from_numpy(np.array(Image.open(scene / f'disp{view}.png'))).float().unsqueeze(0)
                # zero disparities are to be interpreted as inf, set them to nan so they result in nan depth
                disparities[disparities == 0.] = np.nan
                with open(scene / 'dmin.txt') as fh:
                    dmin = int(fh.read().strip())
                # add dmin to disparities because disparity maps and images have been cropped to the joint field of view
                disparities += dmin

                depth_map = baseline * f / disparities
                resize = Resize((int(depth_map.shape[1] * scale), int(depth_map.shape[2] * scale)), scale_interpolation)
                depth_map = resize(depth_map)
                transform = Compose([ToTensor(), resize])
                if use_ambient_images:
                    data.append(
                        [(transform(Image.open(path)), depth_map) for path in
                         scene.glob(f'Illum*/Exp*/view{view}.png')])
                else:
                    data.append([(transform(Image.open(scene / f'view{view}.png')), depth_map)])

    return data


# 2005 dataset same as 2006
def load_2005(*args, **kwargs):
    return load_2006(*args, **kwargs)
