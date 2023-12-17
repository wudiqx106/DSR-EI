import os
import sys
import torch
import numpy as np
from datetime import datetime as dt
# from config import cfg
import torch.nn.functional as F


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# def gradient(dep):
#     n, c, h, w = dep.size()
#
#     x1 = dep[:, :, :, :w - 1]
#     x2 = dep[:, :, :, 1:]
#     Gx = F.pad(x1 - x2, [0, 1, 0, 0, 0, 0], "constant", 0)
#     y1 = dep[:, :, :h - 1, :]
#     y2 = dep[:, :, 1:, :]
#     Gy = F.pad(y1 - y2, [0, 0, 0, 1, 0, 0], 'constant', 0)
#     G = torch.abs(Gx) + torch.abs(Gy)
#
#     return G


def gradient_torch(img, L1=True):
    w, h = img.size(-2), img.size(-1)
    # confidence = torch.sqrt(torch.abs(img[3] - img[1]) + torch.abs(img[0] - img[2]))
    l = F.pad(img, [1, 0, 0, 0])    # left
    r = F.pad(img, [0, 1, 0, 0])    # right
    u = F.pad(img, [0, 0, 1, 0])    # up
    d = F.pad(img, [0, 0, 0, 1])    # down

    if L1:
        return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
    else:
        return torch.sqrt(torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2))


def gradient(img, L1=True):
    # print(img.shape)
    w, h = img.shape[-2], img.shape[-1]
    # confidence = torch.sqrt(torch.abs(img[3] - img[1]) + torch.abs(img[0] - img[2]))
    l = np.pad(img, ((1, 0), (0, 0)))    # left
    r = np.pad(img, ((0, 1), (0, 0)))    # right
    u = np.pad(img, ((0, 0), (1, 0)))    # up
    d = np.pad(img, ((0, 0), (0, 1)))    # down

    if L1:
        return np.abs((l - r)[..., 0:w, 0:h]) + np.abs((u - d)[..., 0:w, 0:h])
    else:
        return np.sqrt(torch.pow((l - r)[..., 0:w, 0:h], 2) + np.pow((u - d)[..., 0:w, 0:h], 2))
