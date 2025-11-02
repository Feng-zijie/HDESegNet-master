"""密集解码器"""
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if num_inputs > 1:
            self.fusion_conv = ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg)
        else:
            self.fusion_conv = None

    def forward(self, inputs):
        
        inputs = [inputs]  # 将 tensor 包装成 list
        inputs = self._transform_inputs(inputs)

        
        # 修改：区分单输入和多输入情况
        if len(inputs) == 1:
            # 单输入情况：直接处理单个特征图
        
            out = self.convs[0](inputs[0])
        
        else:
            outs = []
            for idx in range(len(inputs)):
                x = inputs[idx]
                conv = self.convs[idx]
                outs.append(
                    resize(
                        input=conv(x),
                        size=inputs[0].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners))

            out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out



""""原论文的SegFormer解码头"""
# import torch
# import torch.nn as nn
# from mmcv.cnn import ConvModule

# from mmseg.models.builder import HEADS
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.ops import resize


# @HEADS.register_module()
# class SegformerHead(BaseDecodeHead):
#     """The all mlp Head of segformer.

#     This head is the implementation of
#     `Segformer <https://arxiv.org/abs/2105.15203>` _.

#     Args:
#         interpolate_mode: The interpolate mode of MLP head upsample operation.
#             Default: 'bilinear'.
#     """

#     def __init__(self, interpolate_mode='bilinear', **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)

#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)

#         assert num_inputs == len(self.in_index)

#         self.convs = nn.ModuleList()
#         for i in range(num_inputs):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))

#         self.fusion_conv = ConvModule(
#             in_channels=self.channels * num_inputs,
#             out_channels=self.channels,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)

#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         outs = []
#         for idx in range(len(inputs)):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             outs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners))

#         out = self.fusion_conv(torch.cat(outs, dim=1))

#         out = self.cls_seg(out)

#         return out