from functools import partial
from pathlib import Path

from argdantic import ArgField, ArgParser
from baseg.datamodules import EMSDataModule
from baseg.io import read_raster_profile, write_raster
from baseg.modules import MMSegModule
from baseg.tiling import SmoothTiler
from baseg.utils import find_best_checkpoint, get_experiment_name
from mmengine import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

cli = ArgParser()


@cli.command()
def train(cfg_path: Path = ArgField("-c", description="Path to the config file.")):
    config = Config.fromfile(cfg_path)
    assert "name" in config, "Experiment name not specified in config."
    exp_name = get_experiment_name(config["name"])
    config["name"] = exp_name

    # datamodule
    datamodule = EMSDataModule(
        root=config["data"]["root"],
        patch_size=config["data"]["patch_size"],
        modalities=config["data"]["modalities"],
        batch_size_train=config["data"]["batch_size"],
        batch_size_eval=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # prepare the model
    module = MMSegModule(config["model"])
    logger = TensorBoardLogger(save_dir="outputs", name=exp_name)
    config_dir = Path(logger.log_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config.dump(config_dir / "config.py")
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
        max_epochs=config["trainer"]["max_epochs"],
        precision=config["trainer"]["precision"],
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(module, datamodule=datamodule)


@cli.command()
def test(
    exp_path: Path = ArgField("-e", description="Path to the experiment folder."),
    checkpoint: Path = ArgField(
        "-c",
        default=None,
        description="Path to the checkpoint file. If not specified, the best checkpoint will be loaded.",
    ),
    predict: bool = ArgField(default=False, description="Generate predictions on the test set."),
):
    config_path = exp_path / "config.py"
    models_path = exp_path / "weights"
    assert exp_path.exists(), "Experiment folder does not exist."
    assert config_path.exists(), f"Config file not found in: {config_path}"
    assert models_path.exists(), f"Models folder not found in: {models_path}"
    # load training config
    config = Config.fromfile(config_path)

    # datamodule
    datamodule = EMSDataModule(
        root=config["data"]["root"],
        patch_size=config["data"]["patch_size"],
        modalities=config["data"]["modalities"],
        batch_size_train=config["data"]["batch_size"],
        batch_size_eval=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # prepare the model
    checkpoint = checkpoint or find_best_checkpoint(models_path, "val_loss", "min")
    module_opts = dict(config=config["model"])
    if predict:
        tiler = SmoothTiler(
            tile_size=config["data"]["patch_size"],
            batch_size=config["data"]["batch_size"],
            channels_first=True,
            mirrored=False,
        )
        output_path = exp_path / "predictions"
        output_path.mkdir(parents=True, exist_ok=True)
        inference_fn = partial(process_inference, output_path=output_path)
        module_opts.update(tiler=tiler, predict_callback=inference_fn)
    module = MMSegModule.load_from_checkpoint(checkpoint, **module_opts)

    logger = TensorBoardLogger(save_dir="outputs", name=config["name"], version=exp_path.stem)
    # load the best checkpoint automatically

    if predict:
        trainer = Trainer(**config["evaluation"], logger=False)
        trainer.predict(module, datamodule=datamodule, return_predictions=False)
    else:
        trainer = Trainer(**config["evaluation"], logger=logger)
        trainer.test(module, datamodule=datamodule)


def process_inference(
    batch: dict,
    output_path: Path,
):
    assert output_path.exists(), f"Output path does not exist: {output_path}"
    # for binary segmentation
    prediction = (batch["pred"] > 0.5).int().unsqueeze(0)
    prediction = prediction.cpu().numpy()
    # store the prediction as a GeoTIFF, reading the spatial information from the input image
    image_path = Path(batch["metadata"]["S2L2A"][0])
    input_profile = read_raster_profile(image_path)
    output_profile = input_profile.copy()
    output_profile.update(dtype="uint8", count=1)
    output_file = output_path / f"{image_path.stem}.tif"
    write_raster(path=output_file, data=prediction, profile=output_profile)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli()
