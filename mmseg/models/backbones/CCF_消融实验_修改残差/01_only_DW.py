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

class CrossEncoderFusion(nn.Module):
    def __init__(self):
        super(CrossEncoderFusion, self).__init__()

    def forward(self, x, cnn_encoder_layers, out_indices):
        outs = []
        cnn_encoder_out = x

        # 刚来的 x.shape 为 [16, 3, 256, 256]

        for i, cnn_encoder_layer in enumerate(cnn_encoder_layers):
            # CNN 分支
            cnn_encoder_out = cnn_encoder_layer(cnn_encoder_out)
            
            if i in out_indices:
                outs.append(cnn_encoder_out)

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

        self.cross_encoder_fusion=CrossEncoderFusion()

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

        embed_dims_list = []
        feedforward_channels_list = []

        self.cnn_encoder_layers = nn.ModuleList()
        
        for i, num_layer in enumerate(num_layers): 
            embed_dims_i = embed_dims * num_heads[i]
            embed_dims_list.append(embed_dims_i)
            feedforward_channels_list.append(mlp_ratio * embed_dims_i)

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
                out_indices=self.out_indices
            )
             