import torch
import torch.nn as nn
from timm.models.layers import DropPath
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




from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


'''
CAS-ViT: 用于高效移动应用的卷积加法自注意力视觉变压器 (Arxiv 2024)
即插即用模块：CAS卷积加法自注意力模块（替身模块）
一、背景
Vision Transformers（ViTs）由于其强大的全局上下文建模能力在计算机视觉任务中表现出色，但由于成对令牌
之间的亲和性计算和复杂的矩阵操作，其在资源受限的场景（如移动设备）上的部署面临挑战。虽然以前的研究在减
少计算开销方面取得了一定的进展，但仍不足以解决这些问题。为此，提出了CAS卷积加法自注意力模块，通过简化
令牌混合器，显著降低了计算开销，在效率和性能之间取得了平衡，尤其适用于高效移动视觉应用。

二、CAS卷积加法自注意力模块原理
1. 输入特征：与标准的Transformer注意力机制类似，使用查询（Q）、键（K）和值（V）来表示输入特征。
2. 加法相似函数：引入了一种新的加法相似函数代替传统的乘法操作，减少了计算复杂度，尤其在计算成对的
令牌亲和性时显著提高了效率。
3. 卷积特征增强：通过使用卷积操作增强了局部感知能力，同时利用通道操作和空间操作提取更丰富的特征。
4. 关键模块：
A. 空间操作：使用深度可分离卷积提取空间信息，并通过批量归一化与激活函数进行增强，最后通过sigmoid
函数得到空间注意力权重。
B. 通道操作：使用全局平均池化和卷积操作提取每个通道的重要性，增强通道之间的信息交互。
C. 深度可分离卷积（DWC）：用于计算查询和键融合后的深度特征，进一步降低计算复杂度。
5. 输出特征：经过卷积加法操作后的特征通过投影和丢弃层，最终输出增强后的特征矩阵，有效保留了全局信息
的建模能力，同时显著降低了计算量。
三、适用任务
该模块适用于图像分类、目标检测、实例分割和语义分割等计算机视觉任务。尤其适合资源受限的场景（如移动设
备），在保证计算效率的同时提供具有竞争力的性能。
'''


def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()

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
    
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)
    
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

        # 添加 Spatial 和 Channel 操作模块
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )

        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # RoPE 期望 (H, W, C)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim)) 

    def forward(self, x):
        b, n, c = x.shape
        h_orig, w_orig = self.input_resolution

        h, w = h_orig, w_orig
        if h * w != n:
            # 尝试根据序列长度 N 重新计算 H, W，以适应不同尺度的输入
            h = int(n ** 0.5)
            w = n // h
            if h * w != n:
                raise ValueError(f"Cannot reshape input of length {n} into resolution ({h}, {w}).")
        
        num_heads = self.num_heads
        head_dim = c // num_heads

        # 1. QK 投影
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q_proj, k_proj = qk[0], qk[1]
        v = x 

        # 2. 将 Q 和 K 从 (B, N, C) 转换到 (B, C, H, W)
        q_nchw = q_proj.permute(0, 2, 1).reshape(b, c, h, w)
        k_nchw = k_proj.permute(0, 2, 1).reshape(b, c, h, w)

        # 3. 应用 Spatial 和 Channel 操作
        q_operated_nchw = self.oper_q(q_nchw)
        k_operated_nchw = self.oper_k(k_nchw)

        # 4. 将处理后的 Q 和 K 从 (B, C, H, W) 转换回 (B, N, C)
        q = q_operated_nchw.flatten(2).permute(0, 2, 1)
        k = k_operated_nchw.flatten(2).permute(0, 2, 1)

        # 5. 继续原始 LinearAttention 的逻辑：ELU 激活
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 6. 应用 RoPE (RoPE期望 (B, H, W, C))
        # 确保 RoPE 的输入形状与 RoPE 初始化时的 shape 一致
        # 注意: RoPE 的 `shape` 参数在 `__init__` 中定义为 (H, W, C)，
        # 所以这里的 `reshape` 需要匹配
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # 按照您提供的原始公式：
        z = 1 / (q_rope @ k_rope.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # shape (B, num_heads, N, 1)

        kv = (k_rope.transpose(-2, -1) @ v) * (n ** -1) # (B, num_heads, head_dim, head_dim)

        x = q_rope @ kv * z # (B, num_heads, N, head_dim)

        # Reshape back to (B, N, C)
        x = x.transpose(1, 2).reshape(b, n, c)

        # 8. 应用 LePE
        v_nchw = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v_nchw).permute(0, 2, 3, 1).reshape(b, n, c)
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


# ===== 测试主函数 =====
if __name__ == "__main__":
    # 模拟输入
    B = 16   # batch size
    H = 64  # 高
    W = 64  # 宽
    C = 32  # 通道
    
    x = torch.randn(B, H * W, C)
    
    MLLA_block = MLLA(dim=C, input_resolution=(H, W), num_heads=4, classes=1)
    out = MLLA_block(x)

    print(f"输入: {x.shape}")   
    print(f"输出: {out.shape}")  

