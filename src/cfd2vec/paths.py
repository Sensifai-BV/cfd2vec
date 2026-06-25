"""Data locations without machine-specific paths in the repository.

Every dataset path in configs, scripts and the command line is written with `${CFD2VEC_DATA}`, which defaults to
`~/External/Datasets/cfd`. Set the variable to relocate the data; nothing in the repository needs editing.
"""
from __future__ import annotations

import os

DATA_ENV = "CFD2VEC_DATA"
DEFAULT_DATA_ROOT = "~/External/Datasets/cfd"
CONFIG_PATH_KEYS = ("model", "channel_stats", "shards")


def data_root() -> str:
    """The data root: $CFD2VEC_DATA when set, else the default, with ~ expanded."""
    return os.path.expanduser(os.environ.get(DATA_ENV) or DEFAULT_DATA_ROOT)


def expand(path):
    """Expand `${CFD2VEC_DATA}` (with its default), other environment variables and ~ in a path."""
    if path is None:
        return None
    s = str(path).replace("${" + DATA_ENV + "}", data_root()).replace("$" + DATA_ENV, data_root())
    return os.path.expanduser(os.path.expandvars(s))


def expand_config_paths(y: dict) -> dict:
    """Expand the path entries of a pretraining config in place (`model`, `channel_stats`, `shards`, `manifests`)."""
    for k in CONFIG_PATH_KEYS:
        if y.get(k) is not None:
            y[k] = expand(y[k])
    if y.get("manifests"):
        y["manifests"] = [expand(m) for m in y["manifests"]]
    return y
