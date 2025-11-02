import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 残差连接的“捷径”
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果维度或通道数不匹配，需要对shortcut进行卷积和下采样
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径和捷径相加，然后通过ReLU激活
        return F.relu(self.conv_block(x) + self.shortcut(x), inplace=True)

class ResNetDownsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 初始卷积层，但这里我们直接开始用ResBlock
            DownsamplingResBlock(3, 24, stride=2),      # 256 -> 128
            DownsamplingResBlock(24, 48, stride=2),     # 128 -> 64
            DownsamplingResBlock(48, 64, stride=2),     # 64 -> 32
        )

    def forward(self, x):
        return self.net(x)

# --- 测试模块 ---
if __name__ == '__main__':
    input_tensor = torch.randn(16, 3, 256, 256)
    downsampler = ResNetDownsampler()
    output_tensor = downsampler(input_tensor)
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"ResNet下采样输出形状: {output_tensor.shape}")