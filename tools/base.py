from pathlib import Path
from argdantic import ArgParser, ArgField


cli = ArgParser(name="baseline", description="Baseline training and evaluation script", force_group=True)


@cli.command()
def train(
    root: Path = ArgField(help="Root directory"),
):
    print("Training on root directory", root)


if __name__ == "__main__":
    cli()
