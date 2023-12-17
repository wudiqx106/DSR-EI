from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
from network_utils import gradient
from .utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip

DIML_BASE_SIZE = (756, 1344)


class DIMLDataset256(Dataset):

    def __init__(
            self,
            root='/home/ubuntu/new_disk/qiaoxin/data/NYU_Depthv2',
            resolution='HR',
            # scale=1.0,
            crop_size=(128, 128),
            do_horizontal_flip=True,
            max_rotation_angle=0,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            in_memory=True,
            split='train',
            crop_valid=False,
            crop_deterministic=False,
            scaling=8
    ):
        self.scale = 1
        self.crop_size = (256, 256)
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.scale_interpolation = scale_interpolation
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.crop_valid = crop_valid
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling
        root = Path(root)

        if max_rotation_angle > 0 and crop_deterministic:
            raise ValueError('Max rotation angle has to be zero when cropping deterministically')

        if split not in ('train', 'val', 'test'):
            raise ValueError(split)

        mmap_mode = None if in_memory else 'c'

        self.images = np.load(str(root / f'npy/images_{split}_{resolution}.npy'), mmap_mode)
        self.depth_maps = np.load(str(root / f'npy/depth_{split}_{resolution}.npy'), mmap_mode)
        assert len(self.images) == len(self.depth_maps)

        # self.H, self.W = int(DIML_BASE_SIZE[0] * self.scale), int(DIML_BASE_SIZE[1] * self.scale)
        self.W, self.H = self.images.shape[2:]

        # if self.crop_valid:
        #     if self.max_rotation_angle > 45:
        #         raise ValueError('When crop_valid=True, only rotation angles up to 45° are supported for now')
        #
        #     # make sure that max rotation angle is valid, else decrease
        #     max_angle = np.floor(min(
        #         2 * np.arctan
        #         ((np.sqrt(-(crop_size[0] ** 2) + self.H ** 2 + self.W ** 2) - self.W) / (crop_size[0] + self.H)),
        #         2 * np.arctan
        #         ((np.sqrt(-(crop_size[1] ** 2) + self.W ** 2 + self.H ** 2) - self.H) / (crop_size[1] + self.W))
        #     ) * (180. / np.pi))
        #
        #     if self.max_rotation_angle > max_angle:
        #         print(f'max rotation angle too large for given image size and crop size, decreased to {max_angle}')
        #         self.max_rotation_angle = max_angle

    def __getitem__(self, index):
        if self.crop_deterministic:
            num_crops_h, num_crops_w = self.H // self.crop_size[0], self.W // self.crop_size[1]
            im_index = index // (num_crops_h * num_crops_w)
        else:
            im_index = index

        image = np.array(self.images[im_index], dtype=np.float32).T     # (480, 640, 3) .astype('float32')
        depth_map = np.array(self.depth_maps[im_index], dtype=np.float32).T
        # resize = Resize((self.H, self.W), self.scale_interpolation)
        # image, depth_map = resize(torch.from_numpy(image)), resize(torch.from_numpy(depth_map).unsqueeze(0))
        # image = image.numpy()
        # depth_map = depth_map.squeeze().numpy()

        # if self.do_horizontal_flip and not self.crop_deterministic:
        #     image, depth_map = random_horizontal_flip((image, depth_map))
        #
        # if self.max_rotation_angle > 0  and not self.crop_deterministic:
        #     image, depth_map = random_rotate((image, depth_map), self.max_rotation_angle, self.rotation_interpolation,
        #                                      crop_valid=self.crop_valid)
        #     # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
        #     depth_map[depth_map == 0.] = np.nan

        if self.crop_deterministic:
            crop_index = index % (num_crops_h * num_crops_w)
            crop_index_h, crop_index_w = crop_index // num_crops_w, crop_index % num_crops_w
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            image, depth_map = image[slice_h, slice_w], depth_map[slice_h, slice_w]
        else:
            image, depth_map = random_crop((image, depth_map), self.crop_size)

        # # apply user transforms
        # if self.image_transform is not None:
        #     image = self.image_transform(image)
        # if self.depth_transform is not None:
        #     depth_map = self.depth_transform(depth_map)

        # print('image:', image.shape)

        h, w = image.shape[:2]
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

        # y_bicubic = torch.from_numpy(
        #     bicubic_with_mask(source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
        # y_bicubic = y_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))
        # print(y_bicubic.dtype, image.dtype)

        return {'image': image, 'hr': depth_map, 'mask_hr': mask_hr, 'mask_lr': mask_lr, 'idx': index, 'grad': depth_grad,
                'lr': y_bicubic, 'max': depth_max/10, 'min': depth_min/10}

    def __len__(self):
        if self.crop_deterministic:
            return len(self.depth_maps) * (self.H // self.crop_size[0]) * (self.W // self.crop_size[1])
        return len(self.depth_maps)
