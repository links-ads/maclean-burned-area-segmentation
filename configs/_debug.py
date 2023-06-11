_base_ = [
    "./models/upernet_rn50_aux.py",
    "./datasets/ems.py",
]
name = "_debug"
trainer = dict(
    max_epochs=100,
    accelerator="cpu",
    strategy=None,
    devices=1,
)
