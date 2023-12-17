from pathlib import Path
import json
from network_utils import gradient
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from PIL import Image
from .utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip


class NYUv2Dataset256(Dataset):

    def __init__(
            self,
            root='/home/qiaoxin/prj/Qiao/dataset/NYUDepthv2',
            crop_size=(256, 256),
            do_horizontal_flip=True,
            max_rotation_angle=0,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            in_memory=False,
            split='test',
            crop_valid=True,
            crop_deterministic=True,
            scaling=8,
            **kwargs
    ):
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.crop_valid = crop_valid
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling

        import h5py
        file = h5py.File(Path(root) / 'nyu_depth_v2_labeled.mat')

        with open(Path(root) / 'split_idc.json') as fh:
            self.split_idc = np.array(json.load(fh)[split])

        if max_rotation_angle > 0 and crop_deterministic:
            raise ValueError('Max rotation angle has to be zero when cropping deterministically')

        self.images = np.array(file['images']) if in_memory else file['images']
        self.depth_maps = np.array(file['depths']) if in_memory else file['depths']
        self.instances = np.array(file['instances']) if in_memory else file['instances']
        self.labels = np.array(file['labels']) if in_memory else file['labels']

        self.W, self.H = self.images.shape[2:]

        # if self.crop_valid:
        #     if self.max_rotation_angle > 45:
        #         raise ValueError('When crop_valid=True, only rotation angles up to 45° are supported for now')
        #
        #     # make sure that max rotation angle is valid, else decrease
        #     max_angle = np.floor(min(
        #         2 * np.arctan((np.sqrt(-(crop_size[0] ** 2) + self.H ** 2 + self.W ** 2) - self.W) / (crop_size[0] + self.H)),
        #         2 * np.arctan((np.sqrt(-(crop_size[1] ** 2) + self.W ** 2 + self.H ** 2) - self.H) / (crop_size[1] + self.W))
        #     ) * (180. / np.pi))
        #
        #     if self.max_rotation_angle > max_angle:
        #         print(f'Max rotation angle too large for given image size and crop size, decreased to {max_angle}')
        #         self.max_rotation_angle = max_angle

    def __getitem__(self, index):
        if self.crop_deterministic:
            num_crops_h, num_crops_w = self.H // self.crop_size[0], self.W // self.crop_size[1]
            im_index = self.split_idc[index // (num_crops_h * num_crops_w)]
        else:
            im_index = self.split_idc[index]

        image = self.images[im_index].astype('float32').T
        depth_map = self.depth_maps[im_index].astype('float32').T
        instances = self.instances[im_index].astype('int16').T
        labels = self.labels[im_index].astype('int16').T
        image, depth_map, instances, labels = image.copy(), depth_map.copy(), instances.copy(), labels.copy()

        outputs = [image, depth_map, instances, labels]

        # if self.do_horizontal_flip and not self.crop_deterministic:
        #     outputs = random_horizontal_flip(outputs)
        #
        # if self.max_rotation_angle > 0 and not self.crop_deterministic:
        #     outputs = random_rotate(outputs, self.max_rotation_angle, self.rotation_interpolation,
        #                             crop_valid=self.crop_valid)
        #     # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
        #     outputs[1][outputs[1] == 0.] = np.nan

        if self.crop_deterministic:
            crop_index = index % (num_crops_h * num_crops_w)
            crop_index_h, crop_index_w = crop_index // num_crops_w, crop_index % num_crops_w
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            outputs = [o[slice_h, slice_w] for o in outputs]
        else:
            outputs = random_crop(outputs, self.crop_size)

        # # apply user transforms
        # if self.image_transform is not None:
        #     outputs[0] = self.image_transform(outputs[0])
        # if self.depth_transform is not None:
        #     outputs[1] = self.depth_transform(outputs[1])

        image = outputs[0]
        depth_map = outputs[1]
        # print('depth_map:', depth_map.shape)

        h, w = image.shape[:2]
        # source = downsample(depth_map.unsqueeze(0), self.scaling).squeeze().unsqueeze(0)
        source = np.array(Image.fromarray(depth_map).resize((w//self.scaling, h//self.scaling), Image.BICUBIC)) # bicubic, RMSE=7.13

        # 梯度图
        depth_grad = gradient(depth_map)

        # normalize
        depth_min = np.nanmin(depth_map)
        depth_max = np.nanmax(depth_map)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)   # torch.Size([1, 256, 256])
        source = (source - depth_min) / (depth_max - depth_min)
        depth_grad = depth_grad / (depth_max - depth_min)

        image = image.astype(np.float32).transpose(2, 0, 1) / 255
        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        y_bicubic = np.array(Image.fromarray(source).resize((w, h), Image.BICUBIC))

        source = torch.from_numpy(source).unsqueeze(0).float()
        y_bicubic = torch.from_numpy(y_bicubic).unsqueeze(0).float()
        image = torch.from_numpy(image).float()
        depth_map = torch.from_numpy(depth_map).unsqueeze(0).float()
        depth_grad = torch.from_numpy(depth_grad).unsqueeze(0).float()

        mask_hr = (~torch.isnan(depth_map)).float()
        mask_lr = (~torch.isnan(source)).float()
        mask_grad = (~torch.isnan(depth_grad)).float()
        mask_hr = (mask_hr*mask_grad) # torch.Size([1, 256, 256])

        depth_map[mask_hr == 0.] = 0.
        depth_grad[mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.

        return {'image': image, 'hr': depth_map, 'mask_hr': mask_hr, 'mask_lr': mask_lr, 'idx': index,
                'lr': y_bicubic, 'grad': depth_grad, 'max': depth_max * 100, 'min': depth_min * 100}

    def __len__(self):
        if self.crop_deterministic:
            return len(self.split_idc) * (self.H // self.crop_size[0]) * (self.W // self.crop_size[1])
        return len(self.split_idc)
