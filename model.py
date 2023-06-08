from pathlib import Path
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder
from custom_encoder_decoder import CustomEncoderDecoder
import torch
from torch import nn
import pytorch_lightning as pl



def build_model(config: dict, override: dict = None):
    # build the model from a config file and a checkpoint file
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError("config must be a filename or Config object, " "but got {}".format(type(config)))

    if override is not None:
        config.merge_from_dict(override)
    init_default_scope(config.get("default_scope", "mmseg"))

    model = MODELS.build(config.model)
    model.cfg = config  # save the config in the model for convenience
    return model


config_file = "segformer_cfg.py"

# test a single image and show the results
img = "demo.png"  # or img = mmcv.imread(img), which will only load it once

# override = dict(model=dict(backbone=dict(in_channels=12)))






class SegFormerLightning(pl.Lightningmodule):

    def __init__(self, config_override: dict = None):
        self.super().__init__()

        # override = dict(model=dict(type='CustomEncoderDecoder',
        #                    backbone=dict(in_channels=12),
        #                    decode_head=dict(type='SegformerHead', 
        #                                     num_classes= 1, #TODO
        #                                     loss_decode= dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight= 1.0)), #use_sigmoid = True makes it a BCE loss
        #                    auxiliary_head=dict(type='SegformerHead', 
        #                                     in_channels= [32, 64, 160, 256],
        #                                     in_index= [0, 1, 2, 3],
        #                                     channels= 256,
        #                                     dropout_ratio= 0.1,
        #                                     num_classes= 19,  #TODO
        #                                     norm_cfg= dict(type= 'SyncBN', requires_grad= True),
        #                                     align_corners= False,
        #                                     loss_decode= dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight= 1.0)) # class weights..?
        #                 )
        #         )

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