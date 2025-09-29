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


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

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

class FrequencyAttention(nn.Module):
    def __init__(self, in_dim):
        super(FrequencyAttention, self).__init__()
 
        down_dim = in_dim // 2
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        
        # 添加一个输出投影层来确保维度匹配
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))
 
        self.temperature = nn.Parameter(torch.ones(8, 1, 1))
 
        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
            nn.BatchNorm2d(down_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
            nn.Sigmoid())
 
        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(down_dim)
        self.relu = nn.ReLU(True)
        self.num_heads = 8
 
    def forward(self, x):
        # Convert from (B, N, C) to (B, C, H, W) format
        b, n, c = x.shape
        h = w = int(n ** 0.5)  # Assuming square input
        x_2d = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # B, C, H, W
        
        conv2 = self.conv2(x_2d)
        b, c_conv, h, w = conv2.shape
 
        q_f_2 = torch.fft.fft2(conv2.float())
        k_f_2 = torch.fft.fft2(conv2.float())
        v_f_2 = torch.fft.fft2(conv2.float())
        tepqkv = torch.fft.fft2(conv2.float())
 
        q_f_2 = rearrange(q_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_2 = rearrange(k_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_2 = rearrange(v_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
 
        q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
        k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
        attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature
        attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
        out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
        out_f_2 = rearrange(out_f_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real)*tepqkv))
        out_2 = torch.cat((out_f_2, out_f_l_2), 1)
 
        F_2 = torch.add(out_2, x_2d)
        
        # 使用输出投影确保维度正确
        F_2 = self.output_proj(F_2)
        
        # Convert back to (B, N, C) format
        F_2 = F_2.permute(0, 2, 3, 1).reshape(b, n, c)
        
        return F_2



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
        self.attn = FrequencyAttention(in_dim=dim)
        
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
        x = self.attn(x)

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



def global_median_pooling(x):  #对输入特征图进行全局中值池化操作。

    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled #全局中值池化后的特征图，尺寸为 (batch_size, channels, 1, 1)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        # 定义两个 1x1 卷积层，用于减少和恢复特征维度
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1)) # 全局平均池化
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))# 全局最大池化
        median_pool = global_median_pooling(inputs)# 全局中值池化

        # 处理全局平均池化后的输出
        avg_out = self.fc1(avg_pool)# 通过第一个 1x1 卷积层减少特征维度
        avg_out = F.relu(avg_out, inplace=True) # 应用 ReLU 激活函数
        avg_out = self.fc2(avg_out)# 通过第二个 1x1 卷积层恢复特征维度
        avg_out = torch.sigmoid(avg_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 处理全局最大池化后的输出
        max_out = self.fc1(max_pool)# 通过第一个 1x1 卷积层减少特征维度
        max_out = F.relu(max_out, inplace=True) # 应用 ReLU 激活函数
        max_out = self.fc2(max_out) # 通过第二个 1x1 卷积层恢复特征维度
        max_out = torch.sigmoid(max_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 处理全局中值池化后的输出
        median_out = self.fc1(median_pool) # 通过第一个 1x1 卷积层减少特征维度
        median_out = F.relu(median_out, inplace=True) # 应用 ReLU 激活函数
        median_out = self.fc2(median_out) # 通过第二个 1x1 卷积层恢复特征维度
        median_out = torch.sigmoid(median_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 将三个池化结果的注意力图进行元素级相加
        out = avg_out + max_out + median_out
        return out

"""MECS的空间注意力"""
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # 初始 5x5 深度卷积层
        self.initial_depth_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=5, padding=2, 
            groups=in_channels
        )
        
        # 多个不同尺寸的深度卷积层
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), 
                     padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), 
                     padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), 
                     padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), 
                     padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), 
                     padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), 
                     padding=(10, 0), groups=in_channels),
        ])
        
        # 用于生成空间注意力权重的1x1卷积
        self.attention_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):

        # 先经过 5x5 深度卷积层
        initial_out = self.initial_depth_conv(x)
        
        # 多尺度空间特征提取
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)  # 融合多尺度特征
        
        # 生成空间注意力权重
        spatial_attention = self.attention_conv(spatial_out)
        
        return spatial_attention

##多尺度空洞融合注意力模块。
class MDFA(nn.Module):                       
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, channel_attention_reduce=4):# 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
        super(MDFA, self).__init__()
        self.branch1 = nn.Sequential(# 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，空洞率为6，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=9 * rate, dilation=9 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.channel_attention = ChannelAttention(input_channels=dim_out*5 , internal_neurons=dim_out*5 // channel_attention_reduce)
        self.spatial_attention = SpatialAttention(in_channels=dim_out*5)

    
    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)                     
        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        channel_output=self.channel_attention(feature_cat)
        channel_output=channel_output*feature_cat
        # 最终输出经过降维处理
        spatial_output=self.spatial_attention(feature_cat)
        spatial_output=spatial_output*feature_cat
        
        output = torch.max(channel_output, spatial_output) # 两个模块取显著的值，旨在突出更显著的空间或通道特征
        # output = channel_output + spatial_output
        
        output_cat = output * feature_cat

        result =  self.conv_cat(output_cat)

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
        out = self.layers(x)
        
        out = self.mdfa_block(out)
        
        return out

"""ResNet融合模块"""
class FeaturePyramidFusion(nn.Module):
    def __init__(self, proj_channels=256, num_levels=4):
        super(FeaturePyramidFusion, self).__init__()
        # 类似于ResNet的pyramid_conv和extra_conv，用于投影到统一通道
        # 假设有num_levels层，这里创建ModuleList，但实际使用时根据输入动态
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(proj_channels, proj_channels, kernel_size=1, bias=False)  # 可根据实际通道调整
            for _ in range(num_levels)
        ])
        self.extra_conv = nn.Conv2d(proj_channels, proj_channels, kernel_size=1, bias=False)  # 额外卷积，类似于ResNet

    def forward(self, global_features, local_features):
        # 假设global_features和local_features是当前层的（或列表）
        # 如果是单层：global_features是transformer x (全局)，local_features是cnn_encoder_out (局部)
        # 核心逻辑：投影 -> 上采样全局加到局部 -> 可选降采样 -> 最终加和融合

        # 如果是单个tensor，转为list
        if not isinstance(global_features, list):
            global_features = [global_features]
        if not isinstance(local_features, list):
            local_features = [local_features]

        # 投影到统一通道（类似于ResNet的p4, p3, p2, p1）
        projected = []
        for i, (g, l) in enumerate(zip(global_features, local_features)):
            # 投影全局和局部
            p_g = self.proj_convs[i](g) if g.shape[1] != self.proj_convs[i].in_channels else g  # 动态检查通道
            p_l = self.proj_convs[(i+1) % len(self.proj_convs)](l) if l.shape[1] != self.proj_convs[i].in_channels else l
            
            # 调整分辨率：如果全局分辨率 != 局部，上采样全局到局部大小（类似于ResNet interpolate）
            if p_g.shape[2:] != p_l.shape[2:]:
                p_g = F.interpolate(p_g, size=p_l.shape[2:], mode='nearest')

            # top-down融合：全局加到局部（类似于ResNet p3 = p3 + interpolate(p4)）
            fused_level = p_l + p_g
            projected.append(fused_level)

        # 如果有多层，模拟ResNet的多尺度融合：从高层到低层上采样加和
        if len(projected) > 1:
            for j in range(len(projected)-2, -1, -1):  # 从高层到低层
                upsampled = F.interpolate(projected[j+1], scale_factor=2, mode='nearest')
                if projected[j].shape[2:] != upsampled.shape[2:]:
                    upsampled = F.interpolate(upsampled, size=projected[j].shape[2:], mode='nearest')
                projected[j] = projected[j] + upsampled

            # 可选降采样（类似于ResNet的p2降采样）
            projected[0] = F.interpolate(projected[0], scale_factor=0.5, mode='nearest')

        # 最终融合所有尺度（类似于ResNet fused_feature = p1 + p2 + p3 + ...）
        # 确保分辨率一致后加和
        fused_feature = projected[0]
        for p in projected[1:]:
            p_resized = F.interpolate(p, size=fused_feature.shape[2:], mode='nearest')
            fused_feature = fused_feature + p_resized

        # 额外卷积处理（类似于ResNet的extra_conv）
        fused_feature = self.extra_conv(fused_feature)

        return fused_feature



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
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(cumulative_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        
        self.fusion_blocks = nn.ModuleList([
            FeaturePyramidFusion(proj_channels=out_channels, num_levels=num_layers) 
            for in_channels, growth_rate, num_layers, out_channels in fusion_out_channels
        ])
        
        
        channels_list = [out_channels for _, _, _, out_channels in fusion_out_channels]
        
        self.dense_fusion = DenseFusion(
            in_channels_list=channels_list,
            out_channels=128,
            target_size=(64, 64)  # 可根据需要调整
        )
        
        
    def forward(self, x, cnn_encoder_layers, transformer_encoder_layers, out_indices):
        outs = []
        cnn_encoder_out = x

        # 刚来的 x.shape 为 [16, 3, 256, 256]


        for i, (cnn_encoder_layer, transformer_encoder_layer) in enumerate(zip(cnn_encoder_layers, transformer_encoder_layers)):
            # CNN 分支
            cnn_encoder_out = cnn_encoder_layer(x)
            
            # print("-"*100)
            # print("1",x.shape)
            # print("-"*100)
            
            x, hw_shape = transformer_encoder_layer[0](x)

            # print("-"*100)
            # print("2",x.shape,hw_shape)
            # print("-"*100)

            for block in transformer_encoder_layer[1]:
                x = block(x)
                # print("-"*100)
                # print("3",x.shape,hw_shape)
                # print("-"*100)
                
            x = transformer_encoder_layer[2](x)
            
            # print("-"*100)
            # print("4",x.shape)
            # print("-"*100)
            
            x = nlc_to_nchw(x, hw_shape)
            
            # print("-"*100)
            # print("5",x.shape)
            # print("-"*100)

            # 经过 DenseBlock 进行融合
            x = self.fusion_blocks[i](cnn_encoder_out,x)

        
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
        
        # self.fusion_out_Resnet = [
        #     (32, 16, 3, 32),  
        #     (64, 32, 3, 64),  
        #     (96, 64, 3, 160),
        #     (128, 128, 3, 192)
        # ]

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
            out_indices=self.out_indices
        )
          