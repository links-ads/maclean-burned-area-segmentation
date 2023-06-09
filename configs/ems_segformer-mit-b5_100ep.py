_base_ = [
    "./models/segformer_mit-b5.py",
    "./datasets/ems.py",
]

trainer = dict(
    max_epochs=100,
    precision=16,
    accelerator="gpu",
    strategy="ddp",
    devices=2,
)
