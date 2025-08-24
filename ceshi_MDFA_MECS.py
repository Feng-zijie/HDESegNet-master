import torch
from torch import nn
import torch.nn.functional as F

"""MECS的通道注意力"""
def global_median_pooling(x):  #对输入特征图进行全局中值池化操作。

    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled #全局中值池化后的特征图，尺寸为 (batch_size, channels, 1, 1)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        # 定义两个 1x1 卷积层，用于减少和恢复特征维度
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1)) # 全局平均池化
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))# 全局最大池化
        median_pool = global_median_pooling(inputs)# 全局中值池化

        # 处理全局平均池化后的输出
        avg_out = self.fc1(avg_pool)# 通过第一个 1x1 卷积层减少特征维度
        avg_out = F.relu(avg_out, inplace=True) # 应用 ReLU 激活函数
        avg_out = self.fc2(avg_out)# 通过第二个 1x1 卷积层恢复特征维度
        avg_out = torch.sigmoid(avg_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 处理全局最大池化后的输出
        max_out = self.fc1(max_pool)# 通过第一个 1x1 卷积层减少特征维度
        max_out = F.relu(max_out, inplace=True) # 应用 ReLU 激活函数
        max_out = self.fc2(max_out) # 通过第二个 1x1 卷积层恢复特征维度
        max_out = torch.sigmoid(max_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 处理全局中值池化后的输出
        median_out = self.fc1(median_pool) # 通过第一个 1x1 卷积层减少特征维度
        median_out = F.relu(median_out, inplace=True) # 应用 ReLU 激活函数
        median_out = self.fc2(median_out) # 通过第二个 1x1 卷积层恢复特征维度
        median_out = torch.sigmoid(median_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 将三个池化结果的注意力图进行元素级相加
        out = avg_out + max_out + median_out
        return out

"""MECS的空间注意力"""
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # 初始 5x5 深度卷积层
        self.initial_depth_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=5, padding=2, 
            groups=in_channels
        )
        
        # 多个不同尺寸的深度卷积层
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), 
                     padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), 
                     padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), 
                     padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), 
                     padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), 
                     padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), 
                     padding=(10, 0), groups=in_channels),
        ])
        
        # 用于生成空间注意力权重的1x1卷积
        self.attention_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):

        # 先经过 5x5 深度卷积层
        initial_out = self.initial_depth_conv(x)
        
        # 多尺度空间特征提取
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)  # 融合多尺度特征
        
        # 生成空间注意力权重
        spatial_attention = self.attention_conv(spatial_out)
        
        return spatial_attention

##多尺度空洞融合注意力模块。
class MDFA(nn.Module):                       
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, channel_attention_reduce=4):# 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
        super(MDFA, self).__init__()
        self.branch1 = nn.Sequential(# 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，空洞率为6，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.channel_attention = ChannelAttention(input_channels=dim_out*5 , internal_neurons=dim_out*5 // channel_attention_reduce)
        self.spatial_attention = SpatialAttention(in_channels=dim_out*5)

    
    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        channel_output=self.channel_attention(feature_cat)
        channel_output=channel_output*feature_cat
        # 最终输出经过降维处理
        spatial_output=self.spatial_attention(feature_cat)
        spatial_output=spatial_output*feature_cat
        
        output = torch.max(channel_output, spatial_output)
        
        output_cat = output * feature_cat

        result =  self.conv_cat(output_cat)

        return result


    
if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)  # 随机生成输入数据
    model = MDFA(dim_in=32,dim_out=32,channel_attention_reduce=4)  # 实例化模块
    output = model(input)  # 将输入通过模块处理
    print(output.shape)  # 输出处理后的数据形状