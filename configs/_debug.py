_base_ = [
    "./models/segformer_mit-b0_aux.py",
    "./datasets/ems.py",
]
name = "baseline"
trainer = dict(
    max_epochs=100,
    accelerator="cpu",
    strategy=None,
    devices=None,
)
