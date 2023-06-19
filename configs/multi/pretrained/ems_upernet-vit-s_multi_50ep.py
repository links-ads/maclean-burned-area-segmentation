_base_ = [
    "../../models/upernet_vit-s_aux.py",
    "../../datasets/ems.py",
]
name = "upernet-vit-s_multi_ssl4eo_auxv2_50ep"
trainer = dict(
    max_epochs=50,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
data = dict(
    batch_size_train=32,
    batch_size_eval=32,
    num_workers=8,
)
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
