import math
import torch.nn as nn
import torch
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def convdilated(in_planes, out_planes, stride=1, dilation=1, bn_mom=0.1):
    """3x3 convolution with dilation"""
    padding = dilation
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True)

class SPRModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SPRModule, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        self.fc1 = nn.Conv2d(channels * 5, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        
        out1 = self.avg_pool1(x).view(x.size(0), -1, 1, 1) # 变成[B, channels, 1, 1]
        
        out2 = self.avg_pool2(x).view(x.size(0), -1, 1, 1) # 变成[B, channels×4, 1, 1]
        
        out = torch.cat((out1, out2), 1) # 拼接后 [B, channels + channels×4, 1, 1] = [B, channels×5, 1, 1]
        # 相当于 1个全局（1×1池化）+ 4个局部（2×2池化）= 5个特征
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
    

class MSPAModule(nn.Module):
    def __init__(self, inplanes, scale=3, stride=1, bn_mom=0.1, stype='stage'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            scale: number of scale.
            stride: conv stride.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSPAModule, self).__init__()

        self.width = inplanes
        self.nums = scale
        self.stride = stride
        assert stype in ['stage', 'normal'], 'One of these is suppported (stage or normal)'
        self.stype = stype

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.relu = nn.ModuleList([])

        for i in range(self.nums):
            self.convs.append(convdilated(self.width, self.width, stride=stride, dilation=int((i+1)*3)))

            self.bns.append(nn.BatchNorm2d(self.width))
            self.relu.append(nn.ReLU(inplace=True))
        
        
        self.attention = SPRModule(inplanes * scale)
        
        self.pool_conv = nn.Conv2d(inplanes*scale, inplanes*scale, 1, 1, 0, bias=True) 
        self.pool_bn = nn.BatchNorm2d(inplanes*scale, momentum=bn_mom)
        self.pool_relu = nn.ReLU(inplace=True)    
        
        self.fuse_conv = nn.Conv2d(inplanes * scale * 2, inplanes * scale, 1, 1, 0, bias=False)
        self.fuse_bn   = nn.BatchNorm2d(inplanes * scale)
        self.fuse_relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        [batch_size, c, row, col] = x.size()
        avgpool = x

        spx = torch.split(x, self.width, 1)      
        
        for i in range(self.nums):
            if i == 0 or (self.stype == 'stage' and self.stride != 1):
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relu[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)      
        
        globalpool_feature = self.attention(avgpool)
        
        
        globalpool_feature = F.interpolate(globalpool_feature, (row, col), None, 'bilinear', True)
        
        out = torch.cat((out,globalpool_feature),1)
        
        out = self.fuse_relu(self.fuse_bn(self.fuse_conv(out)))
        
        return out


# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    scale=4
    x = torch.randn(16, 32, 64 ,64)
    
    _,c,_,_=x.shape
    
    MSPA = MSPAModule(inplanes=c//scale,scale=scale)
    # 执行前向传播
    output = MSPA(x)
    # 打印输入和输出的形状
    print('MSPA_input_size:', x.size())
    print('MSPA_output_size:', output.size())
