"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPN implementation
"""

from torchsummary import summary
from model_spn.common import *
from math import pi
# from model_spn.modulated_deform_conv_func import ModulatedDeformConvFunction
# import torch.nn as nn
from mmcv.ops import DeformConv2dPack as DCN
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dFunction
from components.dy_conv import *
from models import MPR_dsr


class NLSPN(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f, n_feat):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time    # number of propagation  18
        self.affinity = self.args.affinity      # affinity type (dynamic pos-neg, dynamic pos, 'static pos-neg, static pos, none'

        self.ch_g = ch_g    # 8
        self.ch_f = ch_f    # 1
        self.k_g = k_g      # 3
        self.k_f = k_f      # 3
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1  # 8
        self.idx_ref = self.num // 2        # 4

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                n_feat, 9 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )       # weight-offset-affinity
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))        # affinity gamma initial multiplier 0.5
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

        self.conf_tmp = DCN(in_channels=1, out_channels=1, kernel_size=3, stride=self.stride,
                            padding=self.padding, deform_groups=self.deformable_groups)

        self.DCN = DCN(in_channels=1, out_channels=1, kernel_size=3, stride=self.stride,
                            padding=self.padding, deform_groups=self.deformable_groups)

        self.act = nn.Sigmoid()
        self.softmax_1 = nn.Softmax(dim=1)
        # dynamic attention
        self.attention = attention2d(n_feat, ratios=0.25, out_planes=4*args.prop_time)
        self.init_dis = [1, 2, 3]

        # adaptive convolution
    #     self.ad_conv = nn.Conv2d(self.ch_g, self.num, kernel_size=(3, 3), padding=1, stride=self.stride)
    #     nn.init.constant_(self.ad_conv.weight, 1)
    #     self.ad_conv.register_full_backward_hook(self._set_lr)
    #
    # @staticmethod
    # def _set_lr(module, grad_input, grad_output):
    #     grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
    #     grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def update_temperature(self):
        self.attention.updata_temperature()

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        global offset, aff
        assert (self.affinity in ['AS', 'ASS', 'TC', 'TGASS']), 'The method has not implemented yet.'

        B, _, H, W = guidance.shape
        offset_aff = self.conv_offset_aff(guidance)
        dy_weight = self.attention(guidance)[:, :, None, None]      # sigmoid

        Aff_all = []
        Offset_all = []
        self.init_dis_len = len(self.init_dis)
        for num in range(self.init_dis_len):
            o1, o2, aff = torch.chunk(offset_aff[:, num*24:(num+1)*24, ...], 3, dim=1)
            o1 = o1 + self.init_dis[num]
            o2 = self.act(o2)
            phase_grid = torch.zeros(o2.shape).type_as(o2)
            for phase in range(self.num):
                phase_grid[:, phase, ...] += phase
            o2 = o2 * pi + phase_grid * 2 * pi * (num + 3) / (3 + self.num)
            x_offset = o1.mul(torch.cos(o2))
            y_offset = o1.mul(torch.sin(o2))
            # ---------------------------------------------------------------
            # Add zero reference offset
            offset = torch.cat((x_offset, y_offset), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:      # absolute-sum Abs−Sum∗
                pass
            elif self.affinity == 'TC':     # Tanh−C
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':      # Tanh−Gamma−Abs−Sum∗
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError

            # Apply confidence
            # TODO : Need more efficient way
            if self.args.conf_prop:
                list_conf = []
                offset_each = torch.chunk(offset, self.num+1, dim=1)

                modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()
                for idx_off in range(0, self.num+1):
                    ww = idx_off % self.k_f
                    hh = idx_off // self.k_f

                    if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                        continue

                    offset_tmp = offset_each[idx_off].detach()

                    # NOTE : Use --legacy option ONLY for the pre-trained models
                    # for ECCV20 results.
                    if self.args.legacy:
                        offset_tmp[:, 0, :, :] = \
                            offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
                        offset_tmp[:, 1, :, :] = \
                            offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

                    # conf_tmp = ModulatedDeformConvFunction.apply(
                    #     confidence, offset_tmp, modulation_dummy, self.w_conf,
                    #     self.b, self.stride, 0, self.dilation, self.groups,
                    #     self.deformable_groups, self.im2col_step)

                    # deformable convolution
                    conf_tmp = ModulatedDeformConv2dFunction.apply(
                        confidence, offset_tmp, modulation_dummy, self.w_conf,
                        self.b, self.stride, 0, self.dilation, self.groups,
                        self.deformable_groups)
                    list_conf.append(conf_tmp)

                conf_aff = torch.cat(list_conf, dim=1)
                # print('conf_aff:', conf_aff.shape)
                aff = aff * conf_aff.contiguous()
                Aff_all.append(aff)
                Offset_all.append(offset)

        Aff_result = []

        for num in range(self.init_dis_len):
            aff = Aff_all[num]
            aff_ref = torch.zeros([B, 1, H, W]).type_as(aff)

            list_aff = list(torch.chunk(aff, self.num, dim=1))
            list_aff.insert(self.idx_ref, aff_ref)
            aff = torch.cat(list_aff, dim=1)

            Aff_result.append(aff)

        return Offset_all, Aff_result, dy_weight

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConv2dFunction.apply(
            feat.float(), offset.float(), aff.float(), self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups)

        return feat

    def aff_normalization(self, Aff_result, dy_weight):
        Aff_abs_all = 0
        for num in range(self.init_dis_len):
            aff = Aff_result[num]
            aff_abs = torch.abs(aff * dy_weight[:, num:num+1, ...])
            aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True)
            Aff_abs_all += aff_abs_sum

        Aff_abs_all += dy_weight[:, self.init_dis_len:self.init_dis_len+1, ...]

        Aff_all = []
        for num in range(self.init_dis_len):
            aff = Aff_result[num] * dy_weight[:, num:num+1, ...] / (Aff_abs_all + 1e-8)
            Aff_all.append(aff)

        return Aff_all, Aff_abs_all

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        global Feat_Result
        assert self.ch_f == feat_init.shape[1]

        if self.args.conf_prop:
            assert confidence is not None

        if self.args.conf_prop:
            offset, aff, dy_weight = self._get_offset_affinity(guidance, confidence)
        else:
            offset, aff, dy_weight = self._get_offset_affinity(guidance, None)

        feat_tmp = feat_init
        list_feat = []
        for k in range(0, self.prop_time):
            Feat_Result = torch.zeros(feat_tmp.shape).type_as(feat_tmp)

            start = 4 * k
            end = 4 * (k + 1)
            aff_prop, Aff_abs_all = self.aff_normalization(aff, dy_weight[:, start:end])
            # print('offset:', len(offset), 'aff_prop:', len(aff_prop))
            for num in range(self.init_dis_len):
                feat_result = self._propagate_once(feat_tmp, offset[num], aff_prop[num])
                Feat_Result += feat_result

            # compute central pixel
            Feat_Result += feat_tmp * dy_weight[:, start + self.init_dis_len:start + self.init_dis_len+1, ...] / Aff_abs_all
            feat_tmp = Feat_Result
            list_feat.append(Feat_Result)

        return Feat_Result, list_feat, offset, aff, self.aff_scale_const.data


class NLSPNModel(nn.Module):
    def __init__(self, args, n_feat=96):
        super(NLSPNModel, self).__init__()

        self.args = args
        self.n_feat = n_feat
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1        # 3*3-1=8

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)

        self.mprnet = MPR_dsr.MPRNet(args, n_feat=n_feat)

        # Init Depth Branch
        # 1/1
        # self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
        #                             padding=1)
        # self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
        #                             padding=1, bn=False, relu=True)

        # Guidance Branch +int(n_feat/2)
        # 1/1
        self.gd_dec1 = conv_bn_relu(n_feat, n_feat, kernel=3, stride=1,
                                    padding=1)
        # self.gd_dec0 = conv_bn_relu(64, 64, kernel=3, stride=1,
        #                             padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(n_feat, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.Sigmoid()
            )

        self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                self.args.prop_kernel, n_feat)

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

        ###############################################
        # Reinitialize weights using He initialization
        ###############################################
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.detach())
        #         m.bias.detach().zero_()
        #     elif isinstance(m, torch.nn.Linear):
        #         nn.init.kaiming_normal_(m.weight.detach())
        #         m.bias.detach().zero_()

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, data):
        rgb, dep = data['image'], data['lr']
        # print(dep.shape)
        # B, C, H, W = dep.shape[0], dep.shape[1], dep.shape[2], dep.shape[3]

        pred, samfeats, EDGE = self.mprnet(data)

        # Init Depth Decoding
        pred_init = pred[0]     # 跳连生成初始深度

        # Guidance Decoding
        guide = self.gd_dec1(samfeats)
        # guide = self.gd_dec0(gd_fd1)
        # gd_fd1 = self.gd_dec1(fe2)
        # guide = self.gd_dec0(gd_fd1)

        if self.args.conf_prop:
            # Confidence Decoding
            # cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            # confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
            cf_fd1 = self.cf_dec1(samfeats)
            confidence = self.cf_dec0(cf_fd1)
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, dep, rgb)

        # Remove negative depth
        y = torch.clamp(y, min=0)
        if self.args.residual_learning:
            y += dep
        results = [y] + pred
        # results = pred

        return results, EDGE


if __name__ == '__main__':
    height = 176
    weight = 240

    # UNet
    # net = UNet().cuda()
    # summary(net, (1, height, weight, 2), batch_size=12)

    # Format
    # net = Format().cuda()
    # summary(net, (1, height, weight), batch_size=12)

    # DepthRefine
    net = NLSPNModel(args)
    summary(net, (4, height, weight), batch_size=1)
