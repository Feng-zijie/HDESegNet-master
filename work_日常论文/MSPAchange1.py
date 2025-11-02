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

        
        avgpool = torch.mean(avgpool, 2, True)
        avgpool = torch.mean(avgpool, 3, True)
        globalpool_feature = self.pool_relu(self.pool_bn(self.pool_conv(avgpool)))
        globalpool_feature = F.interpolate(globalpool_feature, (row, col), None, 'bilinear', True)
        
        out = torch.cat((out,globalpool_feature),1)
        
        # print(globalpool_feature.shape)  # torch.Size([16, 32, 64, 64])
        
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
