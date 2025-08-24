import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
if __name__ == '__main__':
    # 假设输入数据
    batch_size = 4
    channels = 192
    height = 8
    width = 8
    input_tensor1 = torch.randn(batch_size, channels, height, width).cuda()
    input_tensor2 = torch.randn(batch_size, channels, height, width).cuda()

    fusion_module = FeaturePyramidFusion(proj_channels=channels, num_levels=2).cuda()

    # 通过 FeaturePyramidFusion 模块处理输入
    output_tensor = fusion_module(input_tensor1,input_tensor2)

    # 打印输出张量的形状
    print(f"Output shape: {output_tensor.shape}")