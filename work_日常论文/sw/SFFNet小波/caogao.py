import torch
import torch.nn as nn

# =============================================================================
# 1. 依赖模块 (从您提供的 net_torch.py 中提取)
# =============================================================================

class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)

    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2


class Attention(nn.Module):
    def __init__(self, channel, head_channel, dropout):
        super(Attention, self).__init__()
        self.head_channel, self.channel = head_channel, channel
        if channel % head_channel != 0:
            raise ValueError(f"channel ({channel}) must be divisible by head_channel ({head_channel})")
            
        self.q = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.k = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.scale = head_channel ** 0.5
        self.num_head = channel // self.head_channel
        self.mlp_1 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, q, k, v):
        B, q_C, H, W = q.shape
        _, v_C, _, _ = v.shape
        
        # Reshape for Linear layers
        q_flat = q.permute(0, 2, 3, 1).reshape(B, H * W, q_C)
        k_flat = k.permute(0, 2, 3, 1).reshape(B, H * W, q_C)
        v_flat = v.permute(0, 2, 3, 1).reshape(B, H * W, v_C)

        q_attn = self.q(q_flat).reshape(B, H * W, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        k_attn = self.k(k_flat).reshape(B, H * W, self.num_head, self.head_channel).permute(0, 2, 3, 1)
        v_attn_1 = self.v(v_flat)
        v_attn = v_attn_1.reshape(B, H * W, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        
        # Frequency Attention Map
        attn = ((q_attn @ k_attn) / self.scale).softmax(dim=-1)
        
        x = (attn @ v_attn).permute(0, 2, 1, 3).reshape(B, H * W, v_C)
        
        # Reshape back to image format
        v_reshaped = v_attn_1.permute(0, 2, 1).reshape(B, v_C, H, W)
        mlp1_out = self.mlp_1(x).permute(0, 2, 1).reshape(B, v_C, H, W)
        
        rs1 = v_reshaped + mlp1_out
        
        rs1_flat = rs1.permute(0, 2, 3, 1).reshape(B, H * W, v_C)
        mlp2_out = self.mlp_2(rs1_flat).permute(0, 2, 1).reshape(B, v_C, H, W)
        
        rs2 = rs1 + mlp2_out
        
        return rs2

class QKVFuse_new(nn.Module):
    """
    A fusion module for three feature branches (Global, Local, Wavelet)
    inspired by the logic of S_MWiT and Attention.

    Args:
        channel (int): The number of channels for the input and output feature maps.
        head_channel (int): The channel dimension for each attention head.
        dropout (float): The dropout rate.
    """
    def __init__(self, channel, head_channel, dropout=0.1):
        super(QKVFuse_new, self).__init__()
        self.attention = Attention(channel=channel, head_channel=head_channel, dropout=dropout)
        
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        self.resblock = resblock(channel=channel)

    def forward(self, x_global, y_local, z_wavelet):
        """
        Forward pass for the fusion module.

        Args:
            x_global (Tensor): Feature map from the global branch.
            y_local (Tensor): Feature map from the local branch.
            z_wavelet (Tensor): Feature map from the wavelet branch.

        Returns:
            Tensor: The fused feature map.
        """
        # 1. Create the 'Value' by adaptively fusing global and local features.
        # This corresponds to the 'the process of f_v' in the diagram.
        fused_value = self.a * x_global + (1 - self.a) * y_local
        
        attn_output = self.attention(q=z_wavelet, k=z_wavelet, v=fused_value)
        
        final_output = self.resblock(attn_output)
        
        return final_output

# =============================================================================
# 3. 调试和验证代码
# =============================================================================

if __name__ == '__main__':
    # --- 模型参数 ---
    BATCH_SIZE = 16
    CHANNELS = 96      # 三个分支的特征通道数必须相同
    HEAD_CHANNEL = 32  # 每个注意力头的通道数 (CHANNELS 必须能被 HEAD_CHANNEL 整除)
    HEIGHT = 16       # 特征图高度
    WIDTH = 16        # 特征图宽度

    # --- 检查设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 实例化模型 ---
    fusion_model = QKVFuse_new(
            channel=CHANNELS,
            head_channel=HEAD_CHANNEL,
            dropout=0.1).to(device)
    fusion_model.eval() # 设置为评估模式进行测试
    print("\nModel instantiated successfully.")


    x_global = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).to(device)
    y_local = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).to(device)
    z_wavelet = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).to(device)

    print("\n--- Input Shapes ---")
    print(f"Global Branch (x):  {x_global.shape}")
    print(f"Local Branch (y):   {y_local.shape}")
    print(f"Wavelet Branch (z): {z_wavelet.shape}")

    # --- 执行前向传播 ---
    with torch.no_grad(): # 在测试时关闭梯度计算
        fused_output = fusion_model(x_global, y_local, z_wavelet)

    # --- 打印输出形状并验证 ---
    print("\n--- Output Shape ---")
    print(f"Fused Output:       {fused_output.shape}")

    # 验证输出形状是否与输入形状一致
    assert fused_output.shape == x_global.shape, "Output shape does not match input shape!"
    print("\n✅ Verification successful: Output shape is correct.")