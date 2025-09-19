import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

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

# class MLLABlock(nn.Module):
#     def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
#         super().__init__()
        
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio

#         self.norm1 = norm_layer(dim)
#         self.in_proj = nn.Linear(dim, dim)
#         self.act_proj = nn.Linear(dim, dim)
#         self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#         self.act = nn.SiLU()
        
#         # Replace LinearAttention with FrequencyAttention
#         self.attn = MultiScaleMambaAttention(dim=dim,input_resolution=input_resolution,num_heads=num_heads)
        
#         self.out_proj = nn.Linear(dim, dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

#     def forward(self, x, shortcut):
#         H, W = self.input_resolution
#         B, L, C = x.shape

#         act_res = self.act(self.act_proj(x))
#         x = self.in_proj(x).view(B, H, W, C)
#         x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

#         # Frequency Attention (replaces Linear Attention)
#         x = self.attn(x,H,W)

#         x = self.out_proj(x * act_res)
#         x = shortcut + self.drop_path(x)
#         x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

#         return x

class MMLA(nn.Module):
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
        
        self.mla_block = MultiScaleMambaAttention(dim=dim,input_resolution=input_resolution,num_heads=num_heads)
        
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
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        assert L == H * W, "input feature has wrong size"
        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        
        
        x = self.mla_block(x, H, W)
        
        # FFN
        if self.classes:
            x = x + self.drop_path(self.mlp(self.norm2(x))) 
        else: 
            x = x + self.drop_path(self.mix_ffn(self.norm2(x), (H, W)))
            
        return x


# ===== 测试主函数 =====
if __name__ == "__main__":
    # 模拟输入
    B = 16   # batch size
    H = 64  # 高
    W = 64  # 宽
    C = 32  # 通道

    # 随机输入（符合[B, N, C]格式，N=H*W）
    x = torch.randn(B, H * W, C)

    # 创建MM模块
    # mm_block = MultiScaleMambaAttention(dim=C, input_resolution=(H, W), num_heads=8)
    # mlla=MLLABlock


    # # 选择设备
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # mm_block = mm_block.to(device)
    
    # mlla_block = mlla(dim=C, input_resolution=(H, W), num_heads=8).to(device)
    
    # x = x.to(device)

    # # 前向测试
    # out = mm_block(x, H, W)
    
    MMLA_block = MMLA(dim=C, input_resolution=(H, W), num_heads=4, classes=1)
    out = MMLA_block(x)

    print(f"输入: {x.shape}")   
    print(f"输出: {out.shape}")  
