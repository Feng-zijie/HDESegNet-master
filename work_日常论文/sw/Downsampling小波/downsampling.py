import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class WaveletDownsamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, wave='haar', stride=2):
        super().__init__()
        # 对应 WaveletStage 的初始卷积
        self.first_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,stride=stride)

        # --- 主路径组件 ---
        # 1. 对应 DWTTransformer 的小波变换
        self.dwt = DWTForward(J=1, wave=wave)
        # 2. 对应 FeatureEnhancer 的特征增强
        enhancer_in_ch = in_ch * 4 
        self.enhancer_conv1 = nn.Conv2d(enhancer_in_ch, out_ch, kernel_size=3, padding=1)
        self.enhancer_relu = nn.LeakyReLU(inplace=True)
        self.enhancer_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # --- 残差路径组件 ---
        # 对应 ResidualBlock 的残差卷积
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 对应 WaveletStage 的初始卷积
        x = self.first_conv(x)

        # --- 主路径 ---
        # 1. 小波变换 (来自 DWTTransformer.forward)
        yl, yh = self.dwt(x)
        yh0 = yh[0]
        main_path = torch.cat([yl, yh0[:, :, 0], yh0[:, :, 1], yh0[:, :, 2]], dim=1)      
        
        # 2. 特征增强 (来自 FeatureEnhancer.forward)
        main_path = self.enhancer_relu(self.enhancer_conv1(main_path))
        main_path = self.enhancer_relu(self.enhancer_conv2(main_path))

        # --- 残差路径 ---
        residual_path = self.residual_conv(x)

        # --- 路径合并 ---
        # 残差相加 (来自 ResidualBlock.forward)
        out = main_path + residual_path
        return out

# 测试代码
if __name__ == '__main__':
    # 模拟输入（batch=16, channel=96, H=16, W=16）
    input_tensor = torch.randn(16, 160, 16, 16)
    
    # 初始化合并后的降采样模块
    # 输入通道96, 输出通道128
    downsampling_block = WaveletDownsamplingBlock(in_ch=160, out_ch=192,stride=1)
    
    output = downsampling_block(input_tensor)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    # 预期输出：(16, 128, 8, 8) (因stride=2下采样，H和W减半)