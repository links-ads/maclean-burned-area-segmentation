_base_ = [
    "./models/upernet_rn50_aux.py",
    "./datasets/ems.py",
]
name = "upernet-rn50_multi_imnet_100ep"
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
