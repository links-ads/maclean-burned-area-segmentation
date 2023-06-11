import logging
from pathlib import Path

from argdantic import ArgField, ArgParser
from mmengine import Config

from baseg.modules import MultiTaskModule, SingleTaskModule
from baseg.utils import exp_name_timestamp

cli = ArgParser()
log = logging.getLogger("lightning")


@cli.command()
def train(
    cfg_path: Path = ArgField("-c", description="Path to the config file."),
    keep_name: bool = ArgField(
        "-k", default=False, description="Keep the experiment name as specified in the config file."
    ),
):
    log.info(f"Loading config from: {cfg_path}")
    config = Config.fromfile(cfg_path)
    # set the experiment name
    assert "name" in config, "Experiment name not specified in config."
    exp_name = exp_name_timestamp(config["name"]) if not keep_name else config["name"]
    config["name"] = exp_name
    log.info(f"Experiment name: {exp_name}")

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    module_class = MultiTaskModule if "auxiliary_head" in model_config else SingleTaskModule
    module = module_class(model_config)
    ckpt = module.model.state_dict()
    for k in ckpt:
        print(k)


if __name__ == "__main__":
    cli()
