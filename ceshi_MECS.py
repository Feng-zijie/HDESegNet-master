import torch
from torch import nn
import torch.nn.functional as F

class tongdao(nn.Module):  #处理通道部分   函数名就是拼音名称
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准

class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励

class hebing(nn.Module):    #函数名为合并, 意思是把空间和通道分别提取的特征合并起来
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励


class MDFA(nn.Module):                       ##多尺度空洞融合注意力模块。
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):# 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
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
        self.Hebing=hebing(in_channel=dim_out*5)# 整合通道和空间特征的合并模块

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
        larry=self.Hebing(feature_cat)
        larry_feature_cat=larry*feature_cat
        # 最终输出经过降维处理
        result = self.conv_cat(larry_feature_cat)

        return result

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


class MECS(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(MECS , self).__init__()

        self.C = in_channels
        self.O = out_channels
        # 确保输入和输出通道数相同
        assert in_channels == out_channels, "Input and output channels must be the same"
        # 初始化通道注意力模块
        self.channel_attention = ChannelAttention(input_channels=in_channels,
                                                  internal_neurons=in_channels // channel_attention_reduce)

        # 定义 5x5 深度卷积层
        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        # 定义多个不同尺寸的深度卷积层
        self.depth_convs = nn.ModuleList([

            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])
        # 定义 1x1 卷积层和激活函数
        self.pointwise_conv = nn.Conv2d(5*in_channels, in_channels, kernel_size=1, padding=0)
        self.pointwise_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()
        self.mdfa_block=MDFA(dim_in=in_channels, dim_out=out_channels)

    def forward(self, inputs):
        
        inputs = self.mdfa_block(inputs)  # 应用多尺度空洞融合注意力模块
        
        
        # 全局感知机
        inputs = self.pointwise_conv(inputs)
        
        print("2",inputs.shape)  # 打印输入张量的形状
        
        inputs = self.act(inputs)

        # 通道注意力
        channel_att_vec = self.channel_attention(inputs)
        inputs = channel_att_vec * inputs

        # 先经过 5x5 深度卷积层
        initial_out = self.initial_depth_conv(inputs)

        # 空间注意力
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        # 应用空间注意力
        spatial_att = self.pointwise_conv2(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv2(out)
        return out


if __name__ == '__main__':
    # 假设输入数据
    batch_size = 4
    channels = 16
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, channels, height, width).cuda()

    # 初始化 MECS 块
    cpca_block = MECS (in_channels=16, out_channels=16, channel_attention_reduce=4).cuda()

    # 通过 MECS 块处理输入
    output_tensor = cpca_block(input_tensor)

    # 打印输出张量的形状
    print(f"Output shape: {output_tensor.shape}")