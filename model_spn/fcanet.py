import torch.nn as nn
from torchvision.models import ResNet
from model_spn.layer import MultiSpectralAttentionLayer


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FcaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, expansion=1):
        global _mapper_x, _mapper_y
        super(FcaBottleneck, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        if downsample is None:
            expansion = 2
        c2wh = dict([(32, 70), (48, 56), (72, 42), (96, 28), (64, 56), (128, 28), (144, 28), (192, 14), (256, 14), (512, 7)])
        self.planes = planes
        self.inplanes = inplanes
        if self.inplanes != self.planes:
            self.conv_0 = conv1x1(self.inplanes, self.planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes * expansion, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')

        self.downsample = downsample
        if self.downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, self.planes * expansion, stride),
                nn.BatchNorm2d(planes * expansion),
            )
        # else:
        #     self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.stride = stride

    def forward(self, x):
        if self.inplanes != self.planes:
            residual = self.conv_0(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=3,
                 *, reduction=16,):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()

        c2wh = dict([(32, 70), (48, 56), (40, 56), (72, 42), (96, 28), (64, 56), (128, 28), (144, 28), (192, 14), (256, 14), (512, 7)])
        self.planes = planes
        self.inplanes = inplanes
        if self.inplanes != self.planes:
            self.conv_0 = conv1x1(self.inplanes, self.planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method='top16')

        self.stride = stride
        self.downsample = downsample
        if self.downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, self.planes, stride),
                nn.BatchNorm2d(planes),)

    def forward(self, x):
        if self.inplanes != self.planes:
            residual = self.conv_0(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FcaExtractionBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=3,
                 *, reduction=16,):
        global _mapper_x, _mapper_y
        super(FcaExtractionBlock, self).__init__()

        c2wh = dict([(20, 70), (24, 70), (32, 70), (36, 70), (48, 56), (72, 42), (96, 28), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.planes = planes
        self.inplanes = inplanes
        if self.inplanes != self.planes:
            self.conv_0 = conv1x1(self.inplanes, self.planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes], reduction=reduction, freq_sel_method='low8')

        self.stride = stride
        self.is_downsample = downsample
        if self.is_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, self.planes, stride),
                nn.BatchNorm2d(planes),)

    def forward(self, x):
        if self.inplanes != self.planes:
            residual = self.conv_0(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.is_downsample:
            residual = self.downsample(x)

        out -= residual
        out = self.relu(out)

        return out


def fcanet34(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet50(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet101(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet152(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

