import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import numbers

from mim.utils import exit_with_error
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmcv.runner import load_checkpoint

try:
    from mmedit.models.registry import BACKBONES
    from mmedit.utils import get_root_logger
    from mmcv.runner import load_checkpoint
except ImportError:
    exit_with_error('Please install mmedit, mmcv, torch to run this example.')


# import math
import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

class BiAttn(nn.Module):
    '''
    Bi-dimensional attention
    input: [B, H*W, C]
    '''
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


class BiAttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)
        return x


def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class LightViTAttention(nn.Module):
    def __init__(self, dim, num_tokens=1, num_heads=8, window_size=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.global_token = nn.Parameter(torch.zeros(1, self.num_tokens, dim))

        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward_global_aggregation(self, q, k, v):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # @矩阵乘法
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, q, k, v, H, W):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, num_heads, N, C = q.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        # [B, num_heads, H/ws, ws, W/ws, ws, C] -> [(B, H/ws, W/ws), num_heads, ws, ws, C]
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)     # [(B*H/ws*W/ws), num_heads, (ws*ws), C]
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, v.shape[-1])

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)

        # reverse
        x = window_reverse(x, (H, W), (ws, ws))
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward(self, x, H, W):
        # B, N, C = x.shape       # [B,N, C]  N: embeded dim

        global_token = self.global_token.expand(x.shape[0], -1, -1)     # self.num_tokens, embed_dims[0] -> D, C
        # print('global_token:', global_token.shape)
        x1 = torch.cat((global_token, x), dim=1)
        # print('x1:', x1.shape)
        B, N, C = x1.shape
        NT = self.num_tokens
        # qkv
        qkv = self.qkv(x1)
        # .unbind()移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片 切片后：[B, num_heads, N, C]
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)   # 分别切片

        # print('q:', q.shape)
        # split img tokens & global tokens
        x_glb = x[:, :, :NT]
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]
        # x_glb = global_token
        # q_img, k_img, v_img = q, k, v
        # q_glb, _, _ = q, k, v

        # local window attention
        x_img = self.forward_local(q_img, k_img, v_img, H, W)

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img)

        # global broadcast
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        x_img = x_img + self.forward_global_broadcast(q_img, k_glb, v_glb)
        # x = torch.cat([x_glb, x_img], dim=1)
        x = self.proj(x)
        # print('x:', x.shape)
        # x = rearrange(x, 'b (h w) c - > b c h w')

        return x


class TransformerBlock(nn.Module):
    '''
    use LightViTAttention block to reconstruct the high-frequency
    '''
    def __init__(self, dim, num_heads, num_tokens=8, window_size=4, mlp_ratio=8., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention=LightViTAttention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = BiAttnMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x.flatten(2).transpose(1, 2)    # [B, N, C]  N: embeded dim
        # print('x:', x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        return x

################################################################################################################
# LayerNorm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    '''
    不考虑均值，只有方差的normalization
    '''
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    '''
    正常的normalization
    '''
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Gated-Dconv Feed-Forward Network (GDFN)
class GFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BackBoneBlock(nn.Module):
    def __init__(self, num, fm, **args):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm(**args))

    def forward(self, x):
        B, C, H, W = x.shape
        for block in self.arr:
            x = block(x, H, W)
        return x


class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class SCPA(nn.Module):
    """
    DSP module, originated from SCPA
    here, we didn't change the module name
    """
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width // 2, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_d = nn.Conv2d(nf, group_width // 2, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width // 2, group_width // 2, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )

        self.k1_d = nn.Sequential(
            nn.Conv2d(
                group_width // 2, group_width // 2, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )

        self.conv2_a = nn.Conv2d(group_width // 2, group_width, kernel_size=1, bias=False)
        self.conv2_d = nn.Conv2d(group_width // 2, group_width, kernel_size=1, bias=False)

        self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # elastic
        self.down = nn.AvgPool2d(2, stride=2)
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, H, W):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_d = self.down(x)
        out_d = self.conv1_d(out_d)
        out_d = self.lrelu(out_d)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_d = self.k1_d(out_d)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_d = self.lrelu(out_d)

        out_a = self.conv2_a(out_a)
        out_d = self.ups(self.conv2_d(out_d))
        out_a = out_d + out_a

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out


@BACKBONES.register_module()
class SCET(nn.Module):
    def __init__(self, hiddenDim=32, mlpDim=128, n_feat=96):
        super().__init__()
        self.conv3 = nn.Conv2d(4, hiddenDim,
                               kernel_size=3, padding=1)

        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        num_heads = 4
        self.path1 = nn.Sequential(
            BackBoneBlock(16, SCPA, nf=hiddenDim, reduction=2, stride=1, dilation=1),
            BackBoneBlock(1, TransformerBlock,
                          dim=hiddenDim, num_heads=num_heads),
            nn.Conv2d(hiddenDim, n_feat, kernel_size=3, padding=1),
            # nn.PixelShuffle(scaleFactor),
            # nn.Conv2d(hiddenDim,
            #           1, kernel_size=3, padding=1),
        )

        self.path2 = nn.Sequential(
            # nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim,
                      1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1, x2

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


if __name__ == '__main__':
    import torchsummary

    net = SCET(32, 128, 4).cuda()
    torchsummary.summary(net, (3, 48, 48))

