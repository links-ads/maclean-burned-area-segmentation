# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="CustomEncoderDecoder",
    data_preprocessor=None,
    
    backbone=dict(
        type="ResNet",
        pretrained="pretrained/mmseg_rn50_ss4eo.pth",
        depth=50,
        in_channels=13,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    decode_head=dict(
        type="CustomUPerHead",
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
