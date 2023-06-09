from typing import Any, Optional
from torch import nn
import pytorch_lightning as pl
from mmseg.registry import MODELS
from baseg.losses import SoftBCEWithLogitsLoss
from torch.optim import AdamW
from torch.nn import functional as F

class MMSegModule(pl.LightningModule):
    def __init__(self, config: dict, upscaling_dim: Optional[int] = 512):
        super().__init__()
        self.model = MODELS.build(config)
        self.model.cfg = config
        self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255)
        self.criterion_auxiliary = nn.CrossEntropyLoss(ignore_index=255)
        self.upscaling_dim = upscaling_dim
        

    def training_step(self, batch, batch_idx):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + loss_auxiliary

        self.log("train_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_tot", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Any:
        return AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
    
    def validation_step(self, batch, batch_idx):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + loss_auxiliary

        self.log("val_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_tot", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
