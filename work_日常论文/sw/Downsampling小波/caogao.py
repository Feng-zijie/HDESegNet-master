import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

# ==================== 原有模块 (无修改) ====================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, freq_features, spatial_features):
        avg_out = torch.mean(freq_features, dim=1, keepdim=True)
        max_out, _ = torch.max(freq_features, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return spatial_features * attention + freq_features * (1-attention)


class HighFrequencyModule(nn.Module):
    """处理高频特征的模块 (HL, LH, HH)"""
    def __init__(self, in_ch, out_ch):
        super(HighFrequencyModule, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, HL, LH, HH):
        High_frequency = torch.cat([HL, LH, HH], dim=1)
        High_frequency = self.conv_bn_relu(High_frequency)
        High_frequency = self.conv1x1(High_frequency)
        
        return High_frequency 


class LowFrequencyModule(nn.Module):
    """处理低频特征的模块 (LL) - 长条形卷积层"""
    def __init__(self, in_ch, out_ch):
        super(LowFrequencyModule, self).__init__()
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        self.conv1_7 = nn.Conv2d(out_ch, out_ch, (1, 7), padding=(0, 3), groups=out_ch)
        self.conv1_11 = nn.Conv2d(out_ch, out_ch, (1, 11), padding=(0, 5), groups=out_ch)
        self.conv1_21 = nn.Conv2d(out_ch, out_ch, (1, 21), padding=(0, 10), groups=out_ch)
        
        self.conv7_1 = nn.Conv2d(out_ch, out_ch, (7, 1), padding=(3, 0), groups=out_ch)
        self.conv11_1 = nn.Conv2d(out_ch, out_ch, (11, 1), padding=(5, 0), groups=out_ch)
        self.conv21_1 = nn.Conv2d(out_ch, out_ch, (21, 1), padding=(10, 0), groups=out_ch)
        
        self.project_out = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        
    def forward(self, LL):
        LL = self.conv_in(LL)
        
        yL_conv1 = self.conv1_7(LL)
        yL_conv2 = self.conv1_11(LL)
        yL_conv3 = self.conv1_21(LL)
        yL_conv4 = self.conv7_1(LL)
        yL_conv5 = self.conv11_1(LL)
        yL_conv6 = self.conv21_1(LL)
        
        yL_conv = yL_conv1 + yL_conv2 + yL_conv3 + yL_conv4 + yL_conv5 + yL_conv6
        yL_conv = self.project_out(yL_conv)
        
        return yL_conv


# ==================== 新增：小波下采样模块 (无修改) ====================
class WaveletDownsample(nn.Module):
    """
    小波下采样模块，用于阶段之间的过渡
    实现图中的 HDWT + 特征增强 + 残差连接
    """
    def __init__(self, in_ch, out_ch, wave='haar'):
        super(WaveletDownsample, self).__init__()
        
        # 小波变换（自动实现2倍下采样）
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        
        # 主路径：处理小波变换后的4个分量（1个LL + 3个高频）
        # Conv2d's in_channels should be in_ch * 4
        self.main_path = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        
        # 残差路径：3x3卷积 stride=2 下采样
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 主路径：小波变换
        LL, yh = self.dwt(x)  # LL: (B, C, H/2, W/2), yh: [(B, C, 3, H/2, W/2)]
        
        # 拼接4个分量：LL + HL + LH + HH
        HL = yh[0][:, :, 0, :, :]
        LH = yh[0][:, :, 1, :, :]
        HH = yh[0][:, :, 2, :, :]
        fused = torch.cat([LL, HL, LH, HH], dim=1)  # (B, 4C, H/2, W/2)
        
        # 特征增强
        main_out = self.main_path(fused)
        
        # 残差路径
        residual_out = self.residual_path(x)
        
        # 残差相加
        out = self.relu(main_out + residual_out)
        
        return out


# ==================== 修改后的DWT编码层 ====================
class DWTEncoderLayer(nn.Module):
    """单阶段小波编码层（无下采样，仅特征提取）"""
    def __init__(self, in_ch, out_ch):
        super(DWTEncoderLayer, self).__init__()
        # DWT does not change channel count, so in_ch for freq modules is correct
        self.high_freq_module = HighFrequencyModule(in_ch, out_ch)
        self.low_freq_module = LowFrequencyModule(in_ch, out_ch)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x_ll, x_yh):
        # LL and yH are now passed in separately
        LL = x_ll
        yH = x_yh
        
        HL = yH[0][:, :, 0, ::]
        LH = yH[0][:, :, 1, ::]
        HH = yH[0][:, :, 2, ::]
        
        high_freq_out = self.high_freq_module(HL, LH, HH)
        low_freq_out = self.low_freq_module(LL)
        out = self.spatial_attention(high_freq_out, low_freq_out)
        
        return out


# ==================== 测试代码 ====================
if __name__ == "__main__":
    
    # 测试单独的小波下采样模块
    print("\n小波下采样模块测试：")
    downsample = WaveletDownsample(in_ch=64, out_ch=128)
    test_input = torch.randn(16, 64, 128, 128)
    test_output = downsample(test_input)
    print(f"输入 shape: {test_input.shape}")
    print(f"输出 shape: {test_output.shape}")