_base_ = [
    "./models/segformer_mit-b0.py",
    "./datasets/ems.py",
]
name = "baseline"
trainer = dict(
    max_epochs=100,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
