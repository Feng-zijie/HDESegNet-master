# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
warnings.filterwarnings("ignore")
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from torch.nn import Softmax
from typing import Sequence, Optional, Callable
from mmcv.cnn import Conv2d
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList, Sequential

from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from ..utils import make_laplace


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from pytorch_wavelets import DWTForward


class DepthWiseConvModule(BaseModule):
    """An implementation of one Depth-wise Conv Module of LEFormer.

    Args:
        embed_dims (int): The feature dimension.
        feedforward_channels (int): The hidden dimension for FFNs.
        output_channels (int): The output channles of each cnn encoder layer.
        kernel_size (int): The kernel size of Conv2d. Default: 3.
        stride (int): The stride of Conv2d. Default: 2.
        padding (int): The padding of Conv2d. Default: 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(DepthWiseConvModule, self).__init__(init_cfg)
        self.activate = build_activation_layer(act_cfg)
        fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixFFN(nn.Module):
    """An implementation of MixFFN of LEFormer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = nn.GELU()  # 简化激活层构建

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(dropout_layer.get('drop_prob', 0.0)) if dropout_layer else nn.Identity()

    def nlc_to_nchw(self, x, hw_shape):
        """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor."""
        H, W = hw_shape
        assert len(x.shape) == 3
        B, L, C = x.shape
        assert L == H * W
        return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def nchw_to_nlc(self, x):
        """Convert [N, C, H, W] shape tensor to [N, L, C] shape tensor."""
        assert len(x.shape) == 4
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x, hw_shape, identity=None):
        out = self.nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = self.nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)
 
    normalized_tensor = torch.complex(norm_real, norm_imag)
 
    return normalized_tensor

class LinearAttention_B(nn.Module):
    """Linear Attention机制"""
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.input_resolution  # ⬅️ 初始化时已传入

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # 空间降采样
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # 线性 Attention
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        k_cumsum = torch.sum(k, dim=-2, keepdim=True)
        D_inv = 1. / torch.clamp_min(q @ k_cumsum.transpose(-1, -2), 1e-6)

        context = k.transpose(-1, -2) @ v
        attn = (q @ context) * D_inv

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MultiScaleMambaAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, sr_ratio=1, act_layer=nn.SiLU):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.dwconv_3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=False)
        )
        self.dwconv_5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=False)
        )

        self.in_proj2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1)
        self.dwc2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)

        self.act = act_layer()

        self.attn = LinearAttention_B(
            dim=dim // 2,
            input_resolution=input_resolution,   # 初始化时传入
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            sr_ratio=sr_ratio
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, H, W):
        """
        x:  [B, N, C]
        输出: [B, N, C]
        """
        B, L, C = x.shape
        assert L == H * W, "输入序列长度与 H×W 不匹配"

        # --- 激活门控 ---
        act_res = self.act(
            self.act_proj(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .reshape(B, L, C)
        )

        # --- 3x3 分支 ---
        x_s3 = self.dwconv_3(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        x_s3 = self.in_proj2(x_s3).permute(0, 2, 3, 1).reshape(B, L, C // 2)
        x_s3 = self.act(self.dwc2(x_s3.reshape(B, H, W, C // 2).permute(0, 3, 1, 2)))\
                    .permute(0, 2, 3, 1).reshape(B, L, C // 2)
        x_s3 = self.attn(x_s3)

        # --- 5x5 分支 ---
        x_s5 = self.dwconv_5(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        x_s5 = self.in_proj2(x_s5).permute(0, 2, 3, 1).reshape(B, L, C // 2)
        x_s5 = self.act(self.dwc2(x_s5.reshape(B, H, W, C // 2).permute(0, 3, 1, 2)))\
                    .permute(0, 2, 3, 1).reshape(B, L, C // 2)
        x_s5 = self.attn(x_s5)

        # --- 拼接 ---
        x_s = torch.cat((x_s3, x_s5), dim=2)

        # --- 门控融合 + 输出 ---
        x_out = (x_s * act_res).reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        x_out = self.out_proj(x_out)  # 通道融合
        x_out = x_out.permute(0, 2, 3, 1).reshape(B, L, C)  # ⬅️ 还原 [B,N,C]

        return x_out

class MLLABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        
        # Replace LinearAttention with FrequencyAttention
        self.attn = MultiScaleMambaAttention(dim=dim,input_resolution=input_resolution,num_heads=num_heads)
        
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, shortcut):
        H, W = self.input_resolution
        B, L, C = x.shape

        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Frequency Attention (replaces Linear Attention)
        x = self.attn(x,H,W)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        return x


class MLLA(nn.Module):
    r""" MLLA Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, classes=False, **kwargs):
        super().__init__()
        
        self.mlla_block = MLLABlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **kwargs
        )
        
        self.classes = classes
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        if self.classes:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=int(dim * mlp_ratio), 
                act_layer=act_layer, 
                drop=drop
            )
        else:
            self.mix_ffn = MixFFN(
                embed_dims=dim,
                feedforward_channels=4 * dim,
                ffn_drop=drop_path,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path),
                act_cfg=dict(type='GELU')
            )

    def forward(self, x):
        H, W = self.mlla_block.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        
        
        x = self.mlla_block(x, shortcut)
        
        # FFN
        if self.classes:
            x = x + self.drop_path(self.mlp(self.norm2(x))) 
        else: 
            x = x + self.drop_path(self.mix_ffn(self.norm2(x), (H, W)))
            
        return x

class tongdao(nn.Module):  #处理通道部分   函数名就是拼音名称
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=True)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=False)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准

class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=True)  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励

class hebing(nn.Module):    #函数名为合并, 意思是把空间和通道分别提取的特征合并起来
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励


# 修改所有 ReLU 的 inplace 参数为 False
class MDFA(nn.Module):  # 多尺度空洞融合注意力模块
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(MDFA, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False), 
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False), 
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False), 
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=9 * rate, dilation=9 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False), 
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=False)  

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.Hebing = hebing(in_channel=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        larry = self.Hebing(feature_cat)
        larry_feature_cat = larry * feature_cat
        result = self.conv_cat(larry_feature_cat)
        return result
    
class CnnEncoderLayer(BaseModule):
    """Implements one cnn encoder layer in LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            feedforward_channels (int): The hidden dimension for FFNs.
            output_channels (int): The output channles of each cnn encoder layer.
            kernel_size (int): The kernel size of Conv2d. Default: 3.
            stride (int): The stride of Conv2d. Default: 2.
            padding (int): The padding of Conv2d. Default: 0.
            act_cfg (dict): The activation config for FFNs.
                Default: dict(type='GELU').
            ffn_drop (float, optional): Probability of an element to be
                zeroed in FFN. Default 0.0.
            init_cfg (dict, optional): Initialization config dict.
                Default: None.
        """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(CnnEncoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_channels = output_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        self.layers = DepthWiseConvModule(embed_dims=embed_dims,
                                          feedforward_channels=feedforward_channels // 2,
                                          output_channels=output_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          act_cfg=dict(type='GELU'),
                                          ffn_drop=ffn_drop)
        
        self.mdfa_block = MDFA(dim_in=output_channels, dim_out=output_channels)

    def forward(self, x):

        # self.embed_dims => 3, 32, 64, 160   # self.output_channels => 32, 64, 160, 192
        # 第一次 x[16, 3, 256, 256] -> out1[16, 32, 64, 64] -> out2[16, 32, 64, 64]
        # 第二次 x[16, 32, 64, 64] -> out1[16, 64, 32, 32] -> out2[16, 64, 32, 32]
        # 第三次 x[16, 64, 32, 32] -> out1[16, 160, 16, 16] -> out2[16, 160, 16, 16]
        # 第四次 x[16, 160, 16, 16] -> out1[16, 192, 8, 8] -> out2[16, 192, 8, 8]

        out = self.layers(x)
        
        out = self.mdfa_block(out)
        
        return out
    
"""自己加的"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, freq_features , spatial_features):
        # 沿通道做平均池化和最大池化
        avg_out = torch.mean(freq_features, dim=1, keepdim=True)
        max_out, _ = torch.max(freq_features, dim=1, keepdim=True)
        # 拼接后卷积
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        
        # 输出加权后的特征
        return spatial_features * attention

class DWTEncoderLayer(nn.Module): 
    def __init__(self, in_ch, out_ch,s=1):
        super(DWTEncoderLayer, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        self.conv1_7= nn.Conv2d(out_ch, out_ch, (1, 7), padding=(0, 3), groups=out_ch)
        self.conv1_11 = nn.Conv2d(out_ch, out_ch, (1, 11), padding=(0, 5), groups=out_ch)
        self.conv1_21 = nn.Conv2d(out_ch, out_ch, (1, 21), padding=(0, 10), groups=out_ch)
        
        self.conv7_1 = nn.Conv2d(out_ch, out_ch, (7, 1), padding=(3, 0), groups=out_ch)
        self.conv11_1 = nn.Conv2d(out_ch, out_ch, (11, 1), padding=(5, 0), groups=out_ch)
        self.conv21_1 = nn.Conv2d(out_ch, out_ch, (21, 1), padding=(10, 0), groups=out_ch)
        
        self.project_out = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        
        # self.mdfa = MDFA(out_ch , out_ch)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        LL, yH = self.wt(x)
     
        HL = yH[0][:,:,0,::]
        LH = yH[0][:,:,1,::]
        HH = yH[0][:,:,2,::]
        High_frequency = torch.cat([HL, LH, HH], dim=1)
        High_frequency = self.conv_bn_relu(High_frequency)
        LL = self.outconv_bn_relu_L(LL)
        High_frequency = self.outconv_bn_relu_H(High_frequency)
        
        
        yL_conv1=self.conv1_7(LL)
        yL_conv2=self.conv1_11(LL)
        yL_conv3=self.conv1_21(LL)
        yL_conv4=self.conv7_1(LL)
        yL_conv5=self.conv11_1(LL)
        yL_conv6=self.conv21_1(LL)
        
        yL_conv = yL_conv1 + yL_conv2 + yL_conv3 + yL_conv4 + yL_conv5 + yL_conv6
        yL_conv = self.project_out(yL_conv)
        
        out = self.spatial_attention(High_frequency,yL_conv)
    
        return LL , out

class QKVFuse(nn.Module):
    def __init__(self, dim ,num_heads=8):
        super(QKVFuse, self).__init__()
        self.num_heads = num_heads
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, out1, out2):
        
        b, c, h, w = out1.shape
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
            
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        
        # Normalize the queries and keys
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        
        # Attention for out1 and out2
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        
        # Rearrange back to original shape
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out3) + self.project_out(out4) + out1 + out2 # Combine outputs
        
        return out


"""融合模块外进行稠密连接"""
class DenseFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels=64, target_size=(64, 64)):
        super(DenseFusion, self).__init__()
        self.target_size = target_size
        # 为每个阶段创建投影层
        self.proj = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels_list
        ])
        
        # 为稠密连接后的不同通道数创建融合层
        self.fusion_convs = nn.ModuleList()
        cumulative_channels = 0
        for i in range(len(in_channels_list)):
            cumulative_channels += out_channels
            
            # 原本的
            # self.fusion_convs.append(
            #     nn.Sequential(
            #         nn.Conv2d(cumulative_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #         nn.BatchNorm2d(out_channels),
            #         nn.ReLU(inplace=True)
            #     )
            # )
            
            self.fusion_convs.append(
                nn.Sequential(
                    # Depthwise 3×3 卷积（分组数=输入通道数）
                    nn.Conv2d(cumulative_channels, cumulative_channels,
                              kernel_size=3, padding=1, groups=cumulative_channels, bias=False),
                    nn.BatchNorm2d(cumulative_channels),
                    nn.ReLU(inplace=True),
                    # Pointwise 1×1 卷积
                    nn.Conv2d(cumulative_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        # features: list of [N, C, H, W]
        resized = []
        dense_outputs = []
        
        # 投影到统一通道并调整尺寸
        for i, f in enumerate(features):
            f = self.proj[i](f)   # 投影到统一通道
            f = F.interpolate(f, size=self.target_size, mode='bilinear', align_corners=False)
            resized.append(f)
        
        # 逐级稠密拼接并融合
        for i in range(len(resized)):
            if i == 0:
                dense_cat = resized[0]
            else:
                dense_cat = torch.cat([dense_cat, resized[i]], dim=1)
            
            # 通过融合层处理稠密连接的特征
            fused = self.fusion_convs[i](dense_cat)
            dense_outputs.append(fused)

        return dense_outputs[-1]   # 返回最终融合结果


class CrossEncoderFusion(nn.Module):
    def __init__(self, fusion_out_channels):
        super(CrossEncoderFusion, self).__init__()
        
        self.fusion_blocks_two = nn.ModuleList([
            QKVFuse(dim=out_channels) 
            for in_channels, growth_rate, num_layers, out_channels in fusion_out_channels
        ])
        
        channels_list = [out_channels for _, _, _, out_channels in fusion_out_channels]
        
        self.dense_fusion = DenseFusion(
            in_channels_list=channels_list,
            out_channels=128,
            target_size=(64, 64)  # 可根据需要调整
        )
        
        
    def forward(self, x, cnn_encoder_layers, transformer_encoder_layers , dwt_encoder_layers , fusion_conv_layers, out_indices):
        outs = []
        cnn_encoder_out = x
        trans_encoder_out = x

        # 刚来的 x.shape 为 [16, 3, 256, 256]
        
        # 小波层
        LL = x
        dwt_encoder_out = x


        for i, (cnn_encoder_layer, transformer_encoder_layer, dwt_encoder_layer) in enumerate(zip(cnn_encoder_layers, transformer_encoder_layers,dwt_encoder_layers)):
            
            
            LL , dwt_encoder_out = dwt_encoder_layer(LL)
            
            # 局部分支
            cnn_encoder_out = cnn_encoder_layer(x)
            
            # 全局分支
            trans_encoder_out, hw_shape = transformer_encoder_layer[0](x)


            for block in transformer_encoder_layer[1]:
                trans_encoder_out = block(trans_encoder_out)
                
            trans_encoder_out = transformer_encoder_layer[2](trans_encoder_out)
            trans_encoder_out = nlc_to_nchw(trans_encoder_out, hw_shape)
            
            # 经过 DenseBlock 进行融合
            
            out1 = self.fusion_blocks_two[i](dwt_encoder_out,cnn_encoder_out)
            out2 = self.fusion_blocks_two[i](dwt_encoder_out,trans_encoder_out)
            
            x=torch.cat([out1, out2], dim=1)
            x = fusion_conv_layers[i](x)
            
            if i in (0, 1, 2, 3):
                outs.append(x)
        
        if len(outs) > 1:
            dense_out = self.dense_fusion(outs)
            
            return dense_out
        else:

            return outs[0] if outs else x

@BACKBONES.register_module()
class LEFormer(BaseModule):
    """The backbone of LEFormer.

    This backbone is the implementation of `LEFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 32.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [2, 2, 2, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 6].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        pool_numbers (int): the number of Pooling Transformer Layers. Default 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3, # DDD输入图像的通道数，RGB 图像一般是 3。
                 embed_dims=32, # 基础嵌入维度，用于 Transformer 的输入通道数。
                 num_stages=4, # 总共有 4 个阶段，每个阶段可以有不同的 层。
                 num_layers=(2, 2, 2, 2), # 每个阶段的 Transformer 层数。
                 num_heads=(1, 2, 3, 4), # 每个阶段的 Transformer 多头注意力的头数。
                 patch_sizes=(7, 3, 3, 3), # 每个阶段的 Patch Embedding 卷积核大小。
                 dwt_strides=(2, 1, 1, 1),
                 strides=(4, 2, 2, 2), # 每个阶段的 Patch Embedding 步长。
                 sr_ratios=(8, 4, 2, 1), # 每个阶段的 Transformer 编码层的空间缩减率。
                 out_indices=(0, 1, 2, 3), # 注意力缩小比例，用于减少计算量。
                 mlp_ratio=4,  # MLP 隐藏维度与嵌入维度的比例。
                 drop_rate=0.0,
                 ffn_classes=1, # 控制使用 MixFFN 的层数
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(LEFormer, self).__init__(init_cfg=init_cfg)

        self.fusion_out_Resnet = [
            (32, 16, 3, 32),  
            (64, 32, 3, 64),  
            (96, 64, 3, 96),
            (128, 128, 3, 128)
        ]

        self.cross_encoder_fusion=CrossEncoderFusion(self.fusion_out_Resnet)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        cur = 0
        embed_dims_list = []
        feedforward_channels_list = []
        self.transformer_encoder_layers = ModuleList()
        for i, num_layer in enumerate(num_layers):  # num_layer 是每个阶段的 MLLA 层数
            embed_dims_i = embed_dims * num_heads[i]
            embed_dims_list.append(embed_dims_i)
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            feedforward_channels_list.append(mlp_ratio * embed_dims_i)

            
            if embed_dims_i==32:
                self.input_resolution=(64,64)
            elif embed_dims_i==64:
                self.input_resolution=(32,32)
            elif embed_dims_i==96:
                self.input_resolution=(16,16)
            elif embed_dims_i==128:
                self.input_resolution=(8,8)
                
            layer = ModuleList([
                MLLA(
                    dim=embed_dims_i, 
                    input_resolution=self.input_resolution,
                    num_heads=4,
                    classes= i < ffn_classes
                ) 
                    for idx in range(num_layer)
            ])
           
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            
            self.transformer_encoder_layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.cnn_encoder_layers = nn.ModuleList()
        self.dwt_encoder_layers = nn.ModuleList()
        self.fusion_conv_layers = nn.ModuleList()

        for i in range(num_stages):
            self.cnn_encoder_layers.append(
                CnnEncoderLayer(
                    embed_dims=self.in_channels if i == 0 else embed_dims_list[i - 1],
                    feedforward_channels=feedforward_channels_list[i],
                    output_channels=embed_dims_list[i],
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=patch_sizes[i] // 2,
                    ffn_drop=drop_rate
                )
            )
            
            
            self.dwt_encoder_layers.append(
                DWTEncoderLayer(
                    in_ch = self.in_channels if i == 0 else embed_dims_list[i - 1],
                    out_ch = embed_dims_list[i],
                    s = dwt_strides[i]
                )
            )
            
            self.fusion_conv_layers.append(
                Conv2d(
                    in_channels=embed_dims_list[i] * 2,
                    out_channels=embed_dims_list[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
            )
            
            

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LEFormer, self).init_weights()

    def forward(self, x):

        return self.cross_encoder_fusion(
            x,
            cnn_encoder_layers=self.cnn_encoder_layers,
            transformer_encoder_layers=self.transformer_encoder_layers,
            dwt_encoder_layers=self.dwt_encoder_layers,
            fusion_conv_layers=self.fusion_conv_layers,
            out_indices=self.out_indices
        )
          