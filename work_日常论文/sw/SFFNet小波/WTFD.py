import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import torch.nn.functional as F

class WTFDown(nn.Module):#小波变化高低频分解下采样模块
    def __init__(self, in_ch, out_ch):
        super(WTFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        # return yL , yH
        return yL + yH #小波变化高低频分解下采样模块



# """MDFA 模块"""
# class tongdao(nn.Module):  #处理通道部分   函数名就是拼音名称
#     # 通道模块初始化，输入通道数为in_channel
#     def __init__(self, in_channel):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
#         self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=True)  # 1x1卷积用于降维
#         self.relu = nn.ReLU(inplace=False)  # ReLU激活函数，就地操作以节省内存

#     # 前向传播函数
#     def forward(self, x):
#         b, c, _, _ = x.size()  # 提取批次大小和通道数
#         y = self.avg_pool(x)  # 应用自适应平均池化
#         y = self.fc(y)  # 应用1x1卷积
#         y = self.relu(y)  # 应用ReLU激活
#         y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
#         return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准

# class kongjian(nn.Module):
#     # 空间模块初始化，输入通道数为in_channel
#     def __init__(self, in_channel):
#         super().__init__()
#         self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=True)  # 1x1卷积用于产生空间激励
#         self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

#     # 前向传播函数
#     def forward(self, x):
#         y = self.Conv1x1(x)  # 应用1x1卷积
#         y = self.norm(y)  # 应用Sigmoid函数
#         return x * y  # 将空间权重应用到输入x上，实现空间激励

# class hebing(nn.Module):    #函数名为合并, 意思是把空间和通道分别提取的特征合并起来
#     # 合并模块初始化，输入通道数为in_channel
#     def __init__(self, in_channel):
#         super().__init__()
#         self.tongdao = tongdao(in_channel)  # 创建通道子模块
#         self.kongjian = kongjian(in_channel)  # 创建空间子模块

#     # 前向传播函数
#     def forward(self, U):
#         U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
#         U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
#         return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励


# # 修改所有 ReLU 的 inplace 参数为 False
# class MDFA(nn.Module):  # 多尺度空洞融合注意力模块
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(MDFA, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=False), 
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=False), 
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=False), 
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=9 * rate, dilation=9 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=False), 
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=False)  

#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=False),
#         )
#         self.Hebing = hebing(in_channel=dim_out * 5)

#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         larry = self.Hebing(feature_cat)
#         larry_feature_cat = larry * feature_cat
#         result = self.conv_cat(larry_feature_cat)
#         return result


"""自己加的"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, freq_features , spatial_features):
        # 沿通道做平均池化和最大池化
        avg_out = torch.mean(freq_features, dim=1, keepdim=True)
        max_out, _ = torch.max(freq_features, dim=1, keepdim=True)
        # 拼接后卷积
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        
        # 输出加权后的特征
        return spatial_features * attention

class WTFD(nn.Module): 
    def __init__(self, in_ch, out_ch,s=1):
        super(WTFD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        self.conv1_7= nn.Conv2d(out_ch, out_ch, (1, 7), padding=(0, 3), groups=out_ch)
        self.conv1_11 = nn.Conv2d(out_ch, out_ch, (1, 11), padding=(0, 5), groups=out_ch)
        self.conv1_21 = nn.Conv2d(out_ch, out_ch, (1, 21), padding=(0, 10), groups=out_ch)
        
        self.conv7_1 = nn.Conv2d(out_ch, out_ch, (7, 1), padding=(3, 0), groups=out_ch)
        self.conv11_1 = nn.Conv2d(out_ch, out_ch, (11, 1), padding=(5, 0), groups=out_ch)
        self.conv21_1 = nn.Conv2d(out_ch, out_ch, (21, 1), padding=(10, 0), groups=out_ch)
        
        self.project_out = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        
        # self.mdfa = MDFA(out_ch , out_ch)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        yL, yH = self.wt(x)
     
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        
        
        yL_conv1=self.conv1_7(yL)
        yL_conv2=self.conv1_11(yL)
        yL_conv3=self.conv1_21(yL)
        yL_conv4=self.conv7_1(yL)
        yL_conv5=self.conv11_1(yL)
        yL_conv6=self.conv21_1(yL)
        
        yL_conv = yL_conv1 + yL_conv2 + yL_conv3 + yL_conv4 + yL_conv5 + yL_conv6
        yL_conv = self.project_out(yL_conv)
        
        out = self.spatial_attention(yH,yL_conv)
    
        return yH , out

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(16, 32, 64, 64)
    # 创建一个 WTFD实例
    WTFD =  WTFD(32,64,s=1)
    
    # 将输入特征图传递给 WTFD模块
    yH , output = WTFD(input) #小波变化高低频分解模块
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"yH  shape: {yH.shape}")
    print(f"output  shape: {output.shape}")
