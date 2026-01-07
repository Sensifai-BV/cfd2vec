"""Compute configs/channel_stats.json from the training split of the pretraining pool.

    python scripts/channel_stats.py --config configs/pretrain_S_urban.yaml --out configs/channel_stats.json \\
        [--velocity-scale per_channel|common] [--n-per-case 4096] [--seed 0]

The statistics are computed in network space (log1p on k and eps) over present channels of every training case
of the resolved pool (`load_pool` with the config's sources and held-out exclusions), never over validation or
test cases. `--velocity-scale common` gives all three velocity components the horizontal RMS so the velocity part
of the loss is isotropic in physical velocity error; `per_channel` (the historical choice) standardises Uz with its
own standard deviation. The choice, the case count and the pool hash are recorded in the file, so a checkpoint's
`stats` say which objective it was trained with.
"""
import argparse
import json
import sys

import yaml

from cfd2vec.paths import expand_config_paths
from cfd2vec.train.dataset import VELOCITY_SCALES, channel_stats, load_pool, pool_fingerprint, save_json
from cfd2vec.train.pretrain import resolve_manifests


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--velocity-scale", choices=VELOCITY_SCALES, default="per_channel")
    ap.add_argument("--n-per-case", type=int, default=4096); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    y = expand_config_paths(yaml.safe_load(open(a.config)))
    pool = load_pool(resolve_manifests(y), sources=y.get("sources"), exclude_sources=y.get("heldout_sources", []))
    train = pool.loc[pool["split"] == "train", "shard"].tolist()
    st = channel_stats(train, n_per_case=a.n_per_case, seed=a.seed, velocity_scale=a.velocity_scale)
    st.update(pool_sha256=pool_fingerprint(pool), config=a.config)
    save_json(st, a.out)
    print(json.dumps({k: st[k] for k in ("mean", "std", "n_cases", "velocity_scale", "clipped_fraction",
                                          "non_finite_values")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
