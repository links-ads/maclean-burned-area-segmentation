# Burned Area Segmentation
Burned area segmentation from Sentinel-2 using multi-task learning

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
