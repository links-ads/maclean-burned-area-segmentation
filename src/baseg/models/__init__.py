from pathlib import Path
from baseg.models.custom_encoder_decoder import CustomEncoderDecoder
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS

_all__  = ['CustomEncoderDecoder']

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
