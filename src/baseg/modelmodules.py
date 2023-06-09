from pathlib import Path
from baseg.models import build_model

from baseg.models.custom_encoder_decoder import CustomEncoderDecoder
import torch
from torch import nn
import pytorch_lightning as pl

class SegFormerLightning(pl.Lightningmodule):

    def __init__(self, config_file: Path, config_override: dict = None):
        self.super().__init__()

        self.model = build_model(config_file, override=config_override)

        self.criterion_decode= nn.BCEWithLogitsLoss()
        self.criterion_auxiliary = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        decode_out, auxiliary_out = self.model(x)

        loss_decode = self.criterion_decode(decode_out, y_del)
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc)
        loss = loss_decode + loss_auxiliary

        self.log(
            "train_decode_loss", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_auxiliary_loss", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss