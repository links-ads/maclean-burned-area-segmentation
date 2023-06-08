from pathlib import Path
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
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


config_file = "segformer_cfg.py"

# test a single image and show the results
img = "demo.png"  # or img = mmcv.imread(img), which will only load it once
override = dict(model=dict(backbone=dict(in_channels=12)))
model = build_model(config_file, device="cpu", override=override)
model.eval()
# forward stuff
x = torch.randn(1, 12, 512, 512)
y = torch.randint(0, 19, (1, 128, 128))
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

out = model(x)
loss = criterion(out, y)
loss.backward()
optim.step()

print(out)
