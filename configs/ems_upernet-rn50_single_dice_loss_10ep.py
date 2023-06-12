_base_ = [
    "./models/upernet_rn50.py",
    "./datasets/ems.py",
]
name = "upernet-rn50_single_imnet_dice_loss_10ep"
trainer = dict(
    max_epochs=10,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
loss="dice"