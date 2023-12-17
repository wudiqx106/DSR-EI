import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from model_spn.fcanet import FcaExtractionBlock
from models.layers.pasa import Downsample_PASA_group_softmax
from model_spn.layer import res_blocks, upconv
from model_spn import *
from model_spn.components import *
from torchsummary import summary
from models.scet_net import SCET
from thop import profile


class EdgeNet(nn.Module):
    '''
    high-frequency extraction branch
    '''
    def __init__(self, n_feat=96, bias=False):
        super(EdgeNet, self).__init__()
        self.scet = SCET(hiddenDim=32, mlpDim=128)        # (hiddenDim=32, mlpDim=128, scaleFactor=2)
        self.sam = SAM(n_feat=n_feat, kernel_size=3, bias=bias)

    def forward(self, x):
        x1, x2 = self.scet(x)
        x1, hmap = self.sam(x1, x2)

        return x1, hmap


##########################################################################
class BasicConv(nn.Module):
    '''
    conv(transpose conv)-bn-relu
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def conv_cgc(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return Conv2d_CGC(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False, stride=1, pasa_group=2):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        if stride == 2:
            self.pasa = Downsample_PASA_group_softmax(kernel_size=3, stride=stride, in_channels=channel, group=pasa_group)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, (1, 1), padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, (1, 1), padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act=nn.PReLU(), cgc=False):
        super(CAB, self).__init__()
        if cgc:
            self.conv = conv_cgc
        else:
            self.conv = conv
        modules_body = []
        modules_body.append(self.conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(self.conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv_cgc(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        '''
        output:
        x1: features to next stage
        img: hr image
        '''
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


class AFF(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, 3, kernel_size=1, stride=1, relu=True),
            BasicConv(3, 3, kernel_size=3, stride=1, relu=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        # print('n_feat+scale_unetfeats:', n_feat+scale_unetfeats)
        self.conv2 = nn.Conv2d(int(in_channel/3), out_channel, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(n_feat + scale_unetfeats, n_feat, kernel_size=1, stride=1, padding=0)
        self.conv41 = nn.Conv2d(n_feat + scale_unetfeats*2, n_feat, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x4):
        x2 = self.conv21(x2)
        x4 = self.conv41(x4)
        # print('x1:', x1.shape, 'x2:', x2.shape,'x4:', x4.shape,)
        attention = self.conv(torch.cat([x1, x2, x4], dim=1))
        x = attention[:, 0:1, :, :] * x1 + attention[:, 1:2, :, :] * x2 + attention[:, 2:3, :, :] * x4
        x = self.conv2(x)
        return x


class EdgeAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EdgeAttention, self).__init__()
        # C = 96
        # print('in_channel:', in_channel)
        self.cat = conv(in_channel*2, in_channel, kernel_size=3,stride=1)
        # self.res = res_blocks(C, kernel_size=3, num=2)
        # self.res1 = res_blocks(in_channel, kernel_size=3, num=1)
        self.res1 = conv(in_channel, in_channel, kernel_size=3, stride=1)
        self.avgpooling1 = nn.AvgPool2d(3, 1, padding=1)
        self.avgpooling2 = nn.AvgPool2d(3, 1, padding=1)
        self.res2 = res_blocks(in_channel, kernel_size=3, num=2)
        self.sig1 = torch.nn.Sigmoid()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, edge):
        # xx = self.res(x)
        # print('x:', x.shape, 'edge:', edge.shape)
        sum = self.res1(edge)
        sum = self.avgpooling1(sum)
        sum = self.cat(torch.cat([sum, x], 1))
        sum = self.res2(sum)
        sum = self.avgpooling2(sum)
        map = self.sig1(sum)
        y = self.conv(x.mul(map))

        return y


## U-Net
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act, cgc=True) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        self.up_enc1 = GeneralUpSample(n_feat, n_feat)
        self.up_dec1 = GeneralUpSample(n_feat, n_feat)

        self.up_enc2 = GeneralUpSample(n_feat+scale_unetfeats, n_feat+scale_unetfeats)
        self.up_dec2 = GeneralUpSample(n_feat+scale_unetfeats, n_feat+scale_unetfeats)

        self.up_enc3 = GeneralUpSample(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2))
        self.up_dec3 = GeneralUpSample(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2))

        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(self.up_enc1(encoder_outs[0])) + self.csff_dec1(self.up_dec1(decoder_outs[0]))

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(self.up_enc2(encoder_outs[1])) + self.csff_dec2(self.up_dec2(decoder_outs[1]))

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(self.up_enc3(encoder_outs[2])) + self.csff_dec3(self.up_dec3(decoder_outs[2]))

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs, feats):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, feats[1])
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, feats[0])
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor, pasa_group=2, filter_size=1):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(Downsample_PASA_group_softmax(kernel_size=filter_size, stride=2, in_channels=in_channels, group=pasa_group),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class DownSampleInput(nn.Module):
    def __init__(self):
        super(DownSampleInput, self).__init__()
        self.down = nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class GeneralUpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GeneralUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
class HighInfoExtraction(nn.Module):
    '''
    low-cut filtering
    '''
    def __init__(self, in_ch=3, ch_out=64, kernel_size=3, bias=False, reduction=4):
        super(HighInfoExtraction, self).__init__()
        self.conv = FcaExtractionBlock(inplanes=in_ch, planes=ch_out, reduction=reduction, kernel_size=kernel_size, downsample=False)

    def forward(self, x):
        x = self.conv(x)

        return x


##########################################################################
class MPRNet(nn.Module):
    def __init__(self, args=None, in_c=4, out_c=1, n_feat=64, scale_unetfeats=96, kernel_size=3, reduction=4, bias=False):
        super(MPRNet, self).__init__()

        self.args = args
        # downsampling
        self.down = DownSampleInput()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        # HF features extraction
        assert n_feat % 2 == 0, 'An integral number is needed!'
        self.HfreqExtraction_1 = HighInfoExtraction(in_ch=12, ch_out=int(n_feat/2))
        self.HfreqExtraction_2 = HighInfoExtraction(in_ch=3, ch_out=int(n_feat/2))

        self.conv_recon1 = nn.Conv2d(in_channels=n_feat, out_channels=4 * n_feat, kernel_size=kernel_size,
                                     padding=(kernel_size//2))
        self.ps1 = nn.PixelShuffle(2)
        self.conv_recon2 = nn.Conv2d(in_channels=n_feat, out_channels=4 * n_feat, kernel_size=kernel_size,
                                     padding=(kernel_size//2))
        self.ps2 = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(nn.Conv2d(4, int(n_feat/2), kernel_size, bias=bias, padding=(kernel_size//2)),
                                           CAB(int(n_feat/2), kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(nn.Conv2d(1, int(n_feat/2), kernel_size, bias=bias, padding=(kernel_size//2)),
                                           CAB(int(n_feat/2), kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.AFFs = nn.ModuleList([
            AFF(n_feat, scale_unetfeats, n_feat * 3, n_feat),
            AFF(n_feat, scale_unetfeats, n_feat * 3, n_feat),
            AFF(n_feat, scale_unetfeats, n_feat * 3, n_feat),
            AFF(n_feat, scale_unetfeats, n_feat * 3, n_feat),
        ])

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat11 = nn.Sequential(conv(int(n_feat*2), n_feat, kernel_size, bias=bias),
                                      CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias))
        self.concat12 = nn.Sequential(conv(int(n_feat*3), n_feat, kernel_size, bias=bias),
                                      CAB(n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=bias))

        self.edgenet = EdgeNet(n_feat)
        self.EdgeAtt = nn.ModuleList([
            EdgeAttention(n_feat, n_feat),
            EdgeAttention(n_feat, n_feat+scale_unetfeats),
            EdgeAttention(n_feat, n_feat),
            EdgeAttention(n_feat, n_feat+scale_unetfeats),
        ])

    def forward(self, data):
        rgb, dep, grad = data['image'], data['lr'], data['grad']

        # edgenet: to extract the edges in the LR depth
        Fedge1, edge1 = self.edgenet(torch.cat([rgb, dep], dim=1))
        Fedge2 = F.interpolate(Fedge1, scale_factor=0.5)
        Fedge3 = F.interpolate(Fedge2, scale_factor=0.5)
        # -------------------------------------------
        # -------------- Stage 1---------------------
        # -------------------------------------------
        # Compute Shallow Features
        rgb_down2 = self.pixel_unshuffle(rgb)
        dep_down2 = self.pixel_unshuffle(dep)

        # shallow features extraction
        fea1_rgb = self.HfreqExtraction_1(rgb_down2)
        fea1_dep = self.shallow_feat1(dep_down2)

        x1 = self.concat11(torch.cat([fea1_rgb, fea1_dep, Fedge2], dim=1))

        # encoder in stage 1
        feat1 = self.stage1_encoder(x1)     # [enc1, enc2, enc3] res: high->low

        z12 = F.interpolate(feat1[0], scale_factor=0.5)
        z21 = F.interpolate(feat1[1], scale_factor=2)
        z42 = F.interpolate(feat1[2], scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        res2_tmp = self.AFFs[1](z12, feat1[1], z42)  # ch=96
        res2 = self.EdgeAtt[1](res2_tmp, Fedge3)     # 96 96
        res1_tmp = self.AFFs[0](feat1[0], z21, z41)
        res1 = self.EdgeAtt[0](res1_tmp, Fedge2)
        res3 = feat1[2]
        aff1 = [res1, res2, res3]

        # Pass features through Decoder of Stage 1
        res1 = self.stage1_decoder(feat1, aff1)
        up1 = self.ps1(self.conv_recon1(self.act(res1[0])))

        # Apply Supervised Attention Module (SAM)
        x2_samfeats, stage1_img = self.sam12(up1, dep)

        # -------------------------------------------
        # -------------- Stage 2---------------------
        # -------------------------------------------
        # Compute Shallow Features
        fea1_rgb = self.HfreqExtraction_2(rgb)
        fea1_dep = self.shallow_feat2(dep)

        # Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2_cat = self.concat12(torch.cat([fea1_rgb, fea1_dep, x2_samfeats, Fedge1], 1))

        # Process features of both patches with Encoder of Stage 2
        feat2 = self.stage2_encoder(x2_cat, feat1, res1)

        z12 = F.interpolate(feat2[0], scale_factor=0.5)
        z21 = F.interpolate(feat2[1], scale_factor=2)
        z42 = F.interpolate(feat2[2], scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2_tmp = self.AFFs[3](z12, feat2[1], z42)
        res2 = self.EdgeAtt[3](res2_tmp, Fedge2)
        res1_tmp = self.AFFs[2](feat2[0], z21, z41)
        res1 = self.EdgeAtt[2](res1_tmp, Fedge1)
        res3 = feat2[2]
        aff2 = [res1, res2, res3]

        # Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2, aff2)

        # Apply SAM
        samfeats, stage2_img = self.sam23(res2[0], dep)

        EDGE = []
        EDGE.append(edge1)

        if self.args.model == 'NLSPN_mpr':
            return [stage2_img, stage1_img], samfeats, EDGE
        else:
            return [stage2_img, stage1_img], EDGE

if __name__ == '__main__':
    # height = 256
    # weight = 256
    #
    # net = EdgeNet(out_Ch=96).cuda()
    # summary(net, (4, height, weight), batch_size=4)

    model = MPRNet(n_feat=96)
    image = torch.randn(1, 3, 256, 256)  # input shape, batch_size=1
    depth = torch.randn(1, 1, 256, 256)
    depth_hr = torch.randn(1, 1, 256, 256)

    flops, params = profile(model, inputs=([image, depth, depth_hr],))
    print('Flops(G):', flops / 1e9, 'Params(M):', params / 1e6)  # flops/G，para/M
