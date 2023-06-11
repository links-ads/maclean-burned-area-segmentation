_base_ = ["./segformer_mit-b0.py"]
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 18, 3],
        pretrained="pretrained/mmseg-mit_b3.pth",
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
    auxiliary_head=dict(
        type="CustomSegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
)
