# Robust Burned Area Delineation through Multitask Learning

> Dataset coming soon!

## Installation

First, create a python environment (preferably Python 3.10), then install the package itself:
```console
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

Then, install *mmsegmentation*:

```console
$ pip install -U openmim
$ mim install mmengine
$ mim install "mmcv>=2.0.0"
```

At last, you can install `mmseg` by running:

```console
$ pip install "mmsegmentation>=1.0.0"
```

## Data preparation

Make sure the dataset is formatted following the schema in [datasets.py](src/baseg/datasets.py).
By default, the script expects data to be located in `data/ems`.
This can also be achieved by symlinking the dataset to this location:

```console
$ ln -s <absolute_path_to_data> data/ems
```

## Experiments

The experiments exploit a mixture of PyTorch Lightning's and mmsegmentation's logging features to handle most of the experiment's configuration.
Each run creates by default a folder in `output/` with the following structure:

```
outputs/
└── <experiment_name>/
    ├── weights/
    │   ├── <checkpoint_1>.pth
    │   ├── <checkpoint_2>.pth
    │   └── ...
    ├── config.py
    └── tensorboard logs...
```

This folder can later be used to resume training or perform inference.

### Training

Using `mmseg` configuration files (sort of), you can train a model by running:

```console
$ python tools/base.py train -c <config_path>
```

### Testing and Inference

Once a model has been trained, you can test it by running:

```console
# if you just want to compute the metrics
$ python tools/base.py test -e <experiment_path> [-c <checkpoint_path>]
# if you want to save the predictions
$ python tools/base.py test -e <experiment_path> [-c <checkpoint_path>] --predict
```

Where `experiment_path` is the full or relative path to the experiment directory (including the version subdir),
and `checkpoint_path` is the full or relative path to the checkpoint file (including the `.pth` extension).


