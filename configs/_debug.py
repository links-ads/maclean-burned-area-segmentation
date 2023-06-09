_base_ = [
    "./models/segformer_mit-b5_aux.py",
    "./datasets/ems.py",
]
name = "baseline"
trainer = dict(
    max_epochs=100,
    precision=16,
    accelerator="cpu",
    strategy=None,
    devices=None
)
