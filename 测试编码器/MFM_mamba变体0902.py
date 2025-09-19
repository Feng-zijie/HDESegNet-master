import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

# Missing imports - these need to be defined
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LinearAttention_B(nn.Module):
    """Linear Attention mechanism"""
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
        H, W = self.input_resolution
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Linear attention computation
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        k_cumsum = torch.sum(k, dim=-2, keepdim=True)
        D_inv = 1. / (torch.clamp_min(q @ k_cumsum.transpose(-1, -2), 1e-6))
        
        context = k.transpose(-1, -2) @ v
        attn = (q @ context) * D_inv
        
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Softmax(nn.Module):
    """Softmax activation"""
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

# DWT components (simplified versions)
class DWTForward(nn.Module):
    """Discrete Wavelet Transform Forward"""
    def __init__(self, J=1, mode='zero', wave='haar'):
        super().__init__()
        self.J = J
        
    def forward(self, x):
        return x, [x]  # Simplified implementation

class DWTInverse(nn.Module):
    """Discrete Wavelet Transform Inverse"""
    def __init__(self, mode='zero', wave='haar'):
        super().__init__()
        
    def forward(self, x):
        return x[0]  # Simplified implementation

class MFM(nn.Module):
    def __init__(self, dim, out_channel, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, **kwargs):
        super().__init__()

        # 基础参数配置
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # 第一阶段特征处理组件
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 卷积位置编码，提取空间位置信息
        self.norm1 = norm_layer(dim)  # 层归一化
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积投影
        self.in_proj2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1)  # 半维度投影
        self.act_proj = nn.Conv2d(dim, dim, kernel_size=1)  # 激活门控投影
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 深度卷积，提取局部特征
        self.dwc2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)  # 半维度深度卷积

        # 激活函数与注意力机制
        self.act = nn.SiLU()  # SiLU激活函数(f(x) = x * sigmoid(x))
        self.attn_s = LinearAttention_B(  # 完整维度线性注意力
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            sr_ratio=sr_ratio
        )
        
        # 添加半维度注意力机制
        self.attn = LinearAttention_B(  # 半维度线性注意力
            dim=dim // 2,
            input_resolution=input_resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            sr_ratio=sr_ratio
        )

        # 特征输出与正则化
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)  # 输出投影
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath正则化

        # 第二阶段特征处理组件
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 第二层卷积位置编码
        self.norm2 = norm_layer(dim)  # 第二层归一化
        self.mlp = Mlp(  # 多层感知机
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )

        # 输出层
        self.project_out = nn.Conv2d(dim, out_channel, kernel_size=1, bias=False)  # 最终输出投影
        
        # 频域处理组件
        self.norm = nn.BatchNorm2d(dim)  # 批归一化
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')  # 小波变换(未使用)
        self.idwt = DWTInverse(mode='zero', wave='haar')  # 逆小波变换(未使用)

        # 频域特征权重生成器
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),  # 降维压缩
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),  # 升维恢复
            nn.Sigmoid()  # 生成0-1之间的权重
        )

        # 辅助组件
        self.softmax = Softmax(dim=-1)  # Softmax激活
        self.relu = nn.ReLU(True)  # ReLU激活
        
        # 输入输出处理
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, out_channel, 1),  # 初始投影
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))  # 注意力温度参数
        self.ffn = Mlp(dim, dim * 4, dim, act_layer, drop, bias=False)  # 简化版MLP
        self.reduce = nn.Sequential(  # 特征融合层
            nn.Conv2d(out_channel * 2, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

        # 多尺度卷积分支
        self.dwconv_3 = nn.Sequential(  # 3x3卷积分支(小感受野)
            nn.Conv2d(dim, dim, kernel_size=1),  # 投影
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=False)  # 深度卷积
        )

        self.dwconv_5 = nn.Sequential(  # 5x5卷积分支(大感受野)
            nn.Conv2d(dim, dim, kernel_size=1),  # 投影
            nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=False)  # 深度卷积
        )

    def forward(self, x):
        """
        前向传播过程
        参数:
            x: 输入特征图 [B, C, H, W]       
        返回:
            融合后的特征图 [B, out_channel, H, W]
        """
        # 提取输入特征尺寸信息
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # 转换为[B, C, H, W]
        
        
        # 保存初始特征(用于最终残差连接)
        x_0 = self.conv1(x)  # 初始1x1卷积投影
        
        # 特征展平并添加第一阶段位置编码
        x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        x = x + self.cpe1(x.reshape(B, C, H, W)).flatten(2).permute(0, 2, 1)  # 添加卷积位置编码
        shortcut = x  # 保存残差连接
        
        # 频域特征处理
        tepx = torch.fft.fft2(x.reshape(B, H, W, C).permute(0, 3, 1, 2).float())  # 傅里叶变换到频域
        # 应用频域权重并转回空域
        fmt = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(tepx.real) * tepx)))).flatten(2).permute(0, 2, 1)
        
        # 多尺度特征提取
        x_s = self.norm1(x)  # 归一化

        # 3x3卷积分支(小感受野)
        x_s3 = self.dwconv_3(x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # 5x5卷积分支(大感受野)
        x_s5 = self.dwconv_5(x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # 生成激活门控权重
        act_res = self.act(self.act_proj(x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(B, L, C))

        # 处理3x3分支特征
        x_s3 = self.in_proj2(x_s3.reshape(B, H, W, C // 2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, L, C // 2)
        x_s3 = self.act(self.dwc2(x_s3.reshape(B, H, W, C // 2).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C // 2)

        # 处理5x5分支特征
        x_s5 = self.in_proj2(x_s5.reshape(B, H, W, C // 2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, L, C // 2)
        x_s5 = self.act(self.dwc2(x_s5.reshape(B, H, W, C // 2).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C // 2)
        
        # 应用线性注意力机制
        x_s3 = self.attn(x_s3)  # 3x3分支注意力增强 
        x_s5 = self.attn(x_s5)  # 5x5分支注意力增强
        x_s = torch.cat((x_s3, x_s5), 2)  # 拼接多尺度特征

        # 应用激活门控并投影
        x_s = self.out_proj((x_s * act_res).reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(B, L, C)

        # 第一阶段残差连接
        x = shortcut + self.drop_path(x_s) + fmt  # 融合原始特征、多尺度特征和频域特征

        # 第二阶段特征增强
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)  # 添加第二阶段位置编码

        # 再次进行频域特征处理
        tepx = torch.fft.fft2(x.reshape(B, H, W, C).permute(0, 3, 1, 2).float())
        fmt = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(tepx.real) * tepx)))).flatten(2).permute(0, 2, 1)
        
        # 应用FFN(前馈网络)
        x = x + self.drop_path(self.ffn(self.norm2(x))) + fmt  # 融合FFN输出和频域特征

        # 转换回特征图格式并投影到输出维度
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.project_out(x)  # 投影到输出通道数

        # 最终残差连接与特征融合
        x = self.reduce(torch.cat((x_0, x), 1)) + x_0  # 融合初始特征与当前特征
        
        x = x.flatten(2).permute(0, 2, 1)
        
        return x
    
    def extra_repr(self) -> str:
        """返回模块配置信息"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
            f"mlp_ratio={self.mlp_ratio}"

# Test the implementation
if __name__ == "__main__":
    # Test parameters
    batch_size = 16
    dim = 32
    out_channel = dim
    height, width = 64, 64
    input_resolution = (height, width)
    num_heads = 8
    
    # Create model
    model = MFM(
        dim=dim,
        out_channel=out_channel,
        input_resolution=input_resolution,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.1,
        drop_path=0.1
    )
    
    # Test input
    x = torch.randn(batch_size, height*width, dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print("Model test passed!")
    
    
    
