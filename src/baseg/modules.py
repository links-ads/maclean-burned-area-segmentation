from typing import Any, Optional, Callable
from torch import nn
import pytorch_lightning as pl
from mmseg.registry import MODELS
from baseg.losses import SoftBCEWithLogitsLoss
from torch.optim import AdamW
from torch.nn import functional as F
from torchmetrics import F1Score, JaccardIndex
from torch import sigmoid

from mmseg.registry import MODELS
from pytorch_lightning import LightningModule
from torch import nn
import torch
from torch.optim import AdamW

from baseg.losses import SoftBCEWithLogitsLoss


class MMSegModule(LightningModule):
    def __init__(
        self,
        config: dict,
        tiler: Optional[Callable] = None,
        predict_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = MODELS.build(config)
        self.model.cfg = config
        self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255)
        self.criterion_auxiliary = nn.CrossEntropyLoss(ignore_index=255)
        self.f1_del = F1Score(task="binary", average="macro", ignore_index=255)
        self.f1_aux = F1Score(task="multiclass", num_classes=11, average="macro", ignore_index=255)
        self.iou_del = JaccardIndex(task="binary",average="macro", ignore_index=255)
        self.iou_aux = JaccardIndex(task="multiclass", num_classes=11, average="macro", ignore_index=255)
        
        self.tiler = tiler
        self.predict_callback = predict_callback

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + loss_auxiliary

        f1_del = self.f1_del(sigmoid(decode_out.squeeze(1))> 0.5, y_del.float())
        iou_del = self.iou_del(sigmoid(decode_out.squeeze(1))> 0.5, y_del.float())
        f1_aux = self.f1_aux(auxiliary_out.argmax(1), y_lc.long())
        iou_aux = self.iou_aux(auxiliary_out.argmax(1), y_lc.long())

        
        self.log("train_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_del", f1_del, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou_del", iou_del, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_aux", f1_aux, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou_aux", iou_aux, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self) -> Any:
        return AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + loss_auxiliary

        f1_del = self.f1_del(sigmoid(decode_out.squeeze(1))> 0.5, y_del.float())
        iou_del = self.iou_del(sigmoid(decode_out.squeeze(1))> 0.5, y_del.float())
        f1_aux = self.f1_aux(auxiliary_out.argmax(1), y_lc.long())
        iou_aux = self.iou_aux(auxiliary_out.argmax(1), y_lc.long())

        self.log("val_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_del", f1_del, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou_del", iou_del, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_aux", f1_aux, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou_aux", iou_aux, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + loss_auxiliary

        self.log("test_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch["S2L2A"]

        def callback(batch: Any):
            del_out, _ = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predict_callback(batch)
