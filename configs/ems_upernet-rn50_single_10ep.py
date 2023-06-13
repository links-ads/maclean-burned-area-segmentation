_base_ = [
    "./models/upernet_rn50.py",
    "./datasets/ems.py",
]
name = "upernet-rn50_single_imnet_weight5_100ep"
trainer = dict(
    max_epochs=10,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
