from pathlib import Path
from mmengine import Config
from argdantic import ArgParser, ArgField

cli = ArgParser()


@cli.command()
def train(
    cfg_path: Path = ArgField("-c", description="Path to the config file."),
    accelerator: str = ArgField("-a", default="gpu", description="Accelerator to use (see lightning)."),
):
    config = Config.fromfile(cfg_path)
    print(config)


if __name__ == "__main__":
    cli()
