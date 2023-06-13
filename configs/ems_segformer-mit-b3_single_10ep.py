_base_ = [
    "./models/segformer_mit-b3.py",
    "./datasets/ems.py",
]
name = "segformer-mit-b3_single_imnet_weight5_10ep"
trainer = dict(
    max_epochs=10,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
data = dict(
    batch_size_train=32,
    batch_size_eval=32,
    num_workers=4,
)