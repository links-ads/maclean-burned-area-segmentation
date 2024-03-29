import warnings
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torchmetrics import F1Score, JaccardIndex

from baseg.models import build_model


class BaseModule(LightningModule):
    def __init__(
        self,
        config: dict,
        tiler: Optional[Callable] = None,
        predict_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = build_model(config)
        self.model.cfg = config
        self.tiler = tiler
        self.predict_callback = predict_callback
        self.train_metrics = nn.ModuleDict(
            {
                "train_f1": F1Score(task="binary", ignore_index=255, average="macro"),
                "train_iou": JaccardIndex(task="binary", ignore_index=255, average="macro"),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "val_f1": F1Score(task="binary", ignore_index=255, average="macro"),
                "val_iou": JaccardIndex(task="binary", ignore_index=255, average="macro"),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test_f1": F1Score(task="binary", ignore_index=255, average="macro"),
                "test_iou": JaccardIndex(task="binary", ignore_index=255, average="macro"),
            }
        )

    def init_pretrained(self) -> None:
        assert self.model.cfg, "Model config is not set"
        config = self.model.cfg.backbone
        if "pretrained" not in config or config.pretrained is None:
            warnings.warn("No pretrained weights are specified")
            return
        self.model.backbone.load_state_dict(torch.load(config.pretrained), strict=False)   

    def configure_optimizers(self) -> Any:
        return AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
