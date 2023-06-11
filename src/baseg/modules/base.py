from typing import Any, Callable, Optional

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
                "train_f1": F1Score(task="binary", ignore_index=255),
                "train_iou": JaccardIndex(task="binary", ignore_index=255),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "val_f1": F1Score(task="binary", ignore_index=255),
                "val_iou": JaccardIndex(task="binary", ignore_index=255),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test_f1": F1Score(task="binary", ignore_index=255),
                "test_iou": JaccardIndex(task="binary", ignore_index=255),
            }
        )

    def configure_optimizers(self) -> Any:
        return AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
