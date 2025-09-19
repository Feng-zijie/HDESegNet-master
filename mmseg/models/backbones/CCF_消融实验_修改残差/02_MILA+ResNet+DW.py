# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
warnings.filterwarnings("ignore")
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
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

class MixFFN(BaseModule):
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
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
    
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0
        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)
    
class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h, w = self.input_resolution

        # 动态调整 h 和 w，确保 h * w == n
        if h * w != n:
            h = int(n ** 0.5)
            w = n // h
            if h * w != n:
                raise ValueError(f"Cannot reshape input of length {n} into resolution ({h}, {w}).")

        num_heads = self.num_heads
        head_dim = c // num_heads
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 确保 reshape 操作有效
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(b, n, c)

        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
    
class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

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
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, shortcut):
        H, W = self.input_resolution
        B, L, C = x.shape

        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm , classes=False,**kwargs):
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
        else :
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
        
        x=self.mlla_block(x,shortcut)
        
        # FFN
        if self.classes:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else: 
            x = x + self.drop_path(self.mix_ffn(self.norm2(x),(H,W))) 
            
        return x

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
    
    def forward(self, x):

        out = self.layers(x)

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

class CrossEncoderFusion(nn.Module):
    def __init__(self, fusion_out_channels):
        super(CrossEncoderFusion, self).__init__()
        
        self.fusion_blocks = nn.ModuleList([
            FeaturePyramidFusion(proj_channels=out_channels, num_levels=num_layers) 
            for in_channels, growth_rate, num_layers, out_channels in fusion_out_channels
        ])
        
        
    def forward(self, x, cnn_encoder_layers, transformer_encoder_layers, out_indices):
        outs = []
        cnn_encoder_out = x

        for i, (cnn_encoder_layer, transformer_encoder_layer) in enumerate(zip(cnn_encoder_layers, transformer_encoder_layers)):
  
            cnn_encoder_out = cnn_encoder_layer(x)
            
            x, hw_shape = transformer_encoder_layer[0](x)


            for block in transformer_encoder_layer[1]:
                x = block(x)
                
            x = transformer_encoder_layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

            # 经过 ResNet 进行融合
            x = self.fusion_blocks[i](cnn_encoder_out,x)
        
            if i in out_indices:
                outs.append(x)

        return outs

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
                 num_layers=(2, 2, 3, 6), # 每个阶段的 Transformer 层数。
                 num_heads=(1, 2, 5, 6), # 每个阶段的 Transformer 多头注意力的头数。
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
            (160, 64, 3, 160),
            (192, 128, 3, 192)
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
            elif embed_dims_i==160:
                self.input_resolution=(16,16)
            elif embed_dims_i==192:
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