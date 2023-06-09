from pathlib import Path
from baseg.datamodules import EMSDataModule
from baseg.utils import get_experiment_name
from mmengine import Config
from argdantic import ArgParser, ArgField
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.callbacks import ModelCheckpoint


cli = ArgParser()


@cli.command()
def train(cfg_path: Path = ArgField("-c", description="Path to the config file.")):
    config = Config.fromfile(cfg_path)
    assert "name" in config, "Experiment name not specified in config."
    exp_name = get_experiment_name(config["name"])

    # datamodule
    datamodule = EMSDataModule(
        root=config["data"]["root"],
        patch_size=config["data"]["patch_size"],
        modalities=config["data"]["modalities"],
        batch_size_train=config["trainer"]["batch_size"],
        batch_size_eval=config["trainer"]["batch_size"],
        num_workers=config["trainer"]["num_workers"],
    )

    # prepare the model
    module = MMSegModule(config["model"])
    logger = TensorBoardLogger(save_dir="outputs", name=exp_name)
    config.dump(Path(logger.log_dir) / "config.py")
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights",
            monitor="epoch",
            mode="max",
            filename="model-ep-{epoch:02d}_val-loss-{val_loss:.2f}",
            save_top_k=3,
            every_n_epochs=5,
            save_last=True,
        )
    ]
    trainer = Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        strategy=config["trainer"]["strategy"],
        max_epochs=config["trainer"]["epochs"],
        precision=config["trainer"]["precision"],
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    cli()
