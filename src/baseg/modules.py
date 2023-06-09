from torch import nn
import pytorch_lightning as pl
from mmseg.registry import MODELS, init_default_scope
from baseg.losses import SoftBCEWithLogitsLoss


class MMSegModule(pl.Lightningmodule):
    def __init__(self, config: dict):
        self.super().__init__()
        init_default_scope(config.get("default_scope", "mmseg"))
        self.model = MODELS.build(config.model)
        self.model.cfg = config
        self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255)
        self.criterion_auxiliary = nn.CrossEntropyLoss(ignore_index=255)

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

        self.log("train_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_tot", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
