_base_ = [
    "./models/segformer_mit-b3_aux.py",
    "./datasets/ems.py",
]
name = "_debug"
trainer = dict(
    max_epochs=100,
    accelerator="cpu",
    strategy=None,
    devices=1,
)
loss = "dice"
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
