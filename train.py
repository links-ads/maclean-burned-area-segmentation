from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import torch 
import datetime
from lightning.pytorch.loggers import TensorBoardLogger
from model import SegFormerLightning    


pl.seed_everything(42, workers=True)

parser = ArgumentParser()

parser.add_argument(
    "--dataset_path", help="""Path to data folder.""", type=Path
)
parser.add_argument(
    "-s", "--save_path", help="""Path to save model trained model checkpoint."""
)
parser.add_argument(
    "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None
)
parser.add_argument(
    "-n", "--name", help="""Experiment name.""", type=str, default="segformer"
)
parser.add_argument("--num_epochs", help="""Number of Epochs to Run.""", type=int)
parser.add_argument(
        "-amp",
        "--mixed_precision",
        help="""Use mixed precision during training. Defaults to False.""",
        action="store_true",
)
args = parser.parse_args()

config = dict(model=dict(type='CustomEncoderDecoder',
                   backbone=dict(in_channels=12),
                   decode_head=dict(type='SegformerHead', 
                                    num_classes= 1, #TODO
                                    loss_decode= dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight= 1.0)), #use_sigmoid = True makes it a BCE loss
                   auxiliary_head=dict(type='SegformerHead', 
                                    in_channels= [32, 64, 160, 256],
                                    in_index= [0, 1, 2, 3],
                                    channels= 256,
                                    dropout_ratio= 0.1,
                                    num_classes= 19,  #TODO
                                    norm_cfg= dict(type= 'SyncBN', requires_grad= True),
                                    align_corners= False,
                                    loss_decode= dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight= 1.0)) # class weights..?
                )
        )


model = SegFormerLightning(config_override=config)

# data = TODO

save_path = args.save_path if args.save_path is not None else "./models"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=save_path,
    filename= f"{args.name}" + "segformer"+"-{val_loss:.2f}-{val_acc:0.2f}",
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    save_last=True,
)

stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
logger = TensorBoardLogger("lightning_logs_correct", name=f"{args.name}-{datetime.datetime.now()}")
# Instantiate lightning trainer and train model
trainer_args = {
    "accelerator": "gpu" if args.gpus else "cpu",
    "devices": [0] if args.gpus else 1,
    "strategy": "dp" if args.gpus and args.gpus > 1 else "auto",
    "max_epochs": args.num_epochs,
    "callbacks": [checkpoint_callback],
    "precision": 16 if args.mixed_precision else 32,
    "logger": logger
}


trainer = pl.Trainer(**trainer_args)

trainer.fit(model)
trainer.test(model)

# Save trained model weights
torch.save(trainer.model.resnet_model.state_dict(), save_path + f"{args.name}-{args.optimizer}-{args.scheduler}-fold-{k}-{datetime.datetime.now()}-trained_model.pt")