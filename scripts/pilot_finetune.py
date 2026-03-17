"""N-case fine-tuning under the frozen protocol from one initialisation, then test-split evaluation.
In-distribution mechanism check only: the fine-tuning cases come from the urban validation split."""
import argparse
import json
import os

import numpy as np
import torch

from cfd2vec.api import CFD2vec
from cfd2vec.eval.evaluate import evaluate_cases
from cfd2vec.paths import expand
from cfd2vec.train.dataset import shard_paths

ap = argparse.ArgumentParser()
ap.add_argument("--init", help="checkpoint; omit for from-scratch")
ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--N", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tier", default="L"); ap.add_argument("--n-test", type=int, default=12)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--manifest", default="${CFD2VEC_DATA}/cfd2vec_shards/manifest_aero_urban.parquet")
a = ap.parse_args(); a.manifest = expand(a.manifest)
torch.set_num_threads(a.threads)
rng = np.random.default_rng(a.seed)
pool = shard_paths(a.manifest, "val", tiers=[a.tier])
cases = sorted(rng.choice(pool, a.N, replace=False).tolist())
test = shard_paths(a.manifest, "test", tiers=[a.tier])[: a.n_test]
m = CFD2vec.from_pretrained(a.init, device="cpu") if a.init else CFD2vec.from_config("configs/model_pilot.yaml", "configs/channel_stats.json", device="cpu")
os.makedirs(a.out, exist_ok=True)
hist = m.finetune(cases, "configs/finetune_protocol.yaml", seed=a.seed, log_path=os.path.join(a.out, f"{a.name}_N{a.N}_s{a.seed}.json"))
df = evaluate_cases(m.net, test, "cpu", modes=("geometry-only",), baselines=False, seed=a.seed)
df["init"] = a.name; df["N"] = a.N; df["seed"] = a.seed; df["epochs_run"] = len(hist)
df.to_csv(os.path.join(a.out, f"{a.name}_N{a.N}_s{a.seed}.csv"), index=False)
print(a.name, df[[c for c in df.columns if c.startswith("rel_l2")]].median().round(4).to_dict())
