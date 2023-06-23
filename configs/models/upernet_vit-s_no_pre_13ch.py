# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="CustomEncoderDecoder",
    data_preprocessor=None,
    
    backbone=dict(
        type="VisionTransformer",
        img_size=(512, 512),
        patch_size=16,
        in_channels=13,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        final_norm=True,
        with_cls_token=True,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        norm_eval=False,
        interpolate_mode="bicubic",
    ),
    neck=dict(type="MultiLevelNeck", in_channels=[384, 384, 384, 384], out_channels=384, scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type="CustomUPerHead",
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)  # yapf: disable
