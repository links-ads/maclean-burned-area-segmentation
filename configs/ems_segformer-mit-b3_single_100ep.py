_base_ = [
    "./models/segformer_mit-b3.py",
    "./datasets/ems.py",
]
name = "segformer-mit-b3_single_imnet_100ep"
trainer = dict(
    max_epochs=100,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
