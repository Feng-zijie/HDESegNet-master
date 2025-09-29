"""最终的一个stage输出融合特征的配置"""
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LEFormer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2], # 原本的num_layers=[2, 2, 3, 6],
        num_heads=[1, 2, 3, 4],  # 原本的num_heads=[1, 2, 5, 6],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0,1,2,3),  #
        mlp_ratio=4,
        drop_rate=0.0,
        ffn_classes=1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[128],  # 只有一个融合特征，通道数为192
        in_index=[0],       # 只有一个输出索引
        channels=128,       # 可以调整为与输入通道一致或稍小
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            # class_weight=[0.5, 0.5],
            loss_weight=1.0)),
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

"""DenseDecoder + 滑动窗口"""
# """最终的一个stage输出融合特征的配置"""
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     pretrained=None,
#     backbone=dict(
#         type='LEFormer',
#         in_channels=3,
#         embed_dims=32,
#         num_stages=4,
#         num_layers=[2, 2, 2, 2], # 原本的num_layers=[2, 2, 3, 6],
#         num_heads=[1, 2, 3, 4],  # 原本的num_heads=[1, 2, 5, 6],
#         patch_sizes=[7, 3, 3, 3],
#         strides=[4, 2, 2, 2],
#         sr_ratios=[8, 4, 2, 1],
#         out_indices=(3,),  # 只输出最后一个融合特征，而不是四个阶段
#         mlp_ratio=4,
#         drop_rate=0.0,
#         ffn_classes=1),
#     decode_head=dict(
#         type='SegformerHead',
#         in_channels=[128],  # 只有一个融合特征，通道数为192
#         in_index=[0],       # 只有一个输出索引
#         channels=128,       # 可以调整为与输入通道一致或稍小
#         dropout_ratio=0.1,
#         num_classes=3,
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=False,
#             # class_weight=[0.5, 0.5],
#             loss_weight=1.0)),
    
#     train_cfg=dict(),
#     # test_cfg=dict(mode='whole'))
    
    
#     # 滑动窗口推理配置（关键修改）
#     test_cfg=dict(
#         mode='slide',
#         crop_size=(416, 416),     # 与数据配置中crop_size一致
#         stride=(256, 256)         # 推荐2/3重叠
#     )
# )

# find_unused_parameters=True