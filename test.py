from pathlib import Path
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder
from baseg.models.custom_encoder_decoder import CustomEncoderDecoder
import torch
from torch import nn


def build_model(config: dict, device: str = "cpu", override: dict = None):
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
    model.to(device)
    return model


config_file = "configs/segformer_cfg.py"

# test a single image and show the results
img = "demo.png"  # or img = mmcv.imread(img), which will only load it once

# override = dict(model=dict(backbone=dict(in_channels=12)))



override = dict(model=dict(type='CustomEncoderDecoder',
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

model = build_model(config_file, device="cpu", override=override)
model.eval()
# forward stuff
x = torch.randn(1, 12, 512, 512)
y = torch.randint(0, 19, (1, 128, 128))
y_binary = torch.randint(0, 2, (1, 128, 128)).unsqueeze(1).float()
print(y.shape)
print(y_binary.shape)
criterion_decode= nn.BCEWithLogitsLoss()
criterion_auxiliary = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# out = model(x, mode="loss")
decode_out, auxiliary_out = model(x)
loss_decode = criterion_decode(decode_out, y_binary)
loss_auxiliary = criterion_auxiliary(auxiliary_out, y)
(loss_decode + loss_auxiliary).backward()
optim.step()

