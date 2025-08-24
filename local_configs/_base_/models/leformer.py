norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LEFormer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 3, 6],
        num_heads=[1, 2, 5, 6],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), # 表示输出的特征图，返回四个stage
        mlp_ratio=4,
        drop_rate=0.0,
        ffn_classes=1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 192], # 对应地告诉解码器，这些特征的 channel 数分别是多少
        in_index=[0, 1, 2, 3], # 对应backbone中的out_indices=(0, 1, 2, 3)，返回的四个阶段结果
        channels=192, 
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