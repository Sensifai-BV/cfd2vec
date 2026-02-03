"""Full-field test-split evaluation of a pretrained checkpoint (geometry-only and with-prior modes, plus baselines)."""
import argparse
import os

import torch

from cfd2vec.api import CFD2vec
from cfd2vec.eval.evaluate import evaluate_cases
from cfd2vec.paths import expand
from cfd2vec.train.dataset import shard_paths

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--manifest", default="${CFD2VEC_DATA}/cfd2vec_shards/manifest_aero_urban.parquet")
ap.add_argument("--split", default="test"); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--no-baselines", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
m = CFD2vec.from_pretrained(a.ckpt, device="cpu")
paths = shard_paths(expand(a.manifest), a.split)
df = evaluate_cases(m.net, paths, "cpu", baselines=not a.no_baselines)
df["checkpoint"] = os.path.basename(os.path.dirname(a.ckpt)); df["objective"] = m.meta.get("objective"); df["step"] = m.meta.get("step")
df.to_csv(a.out, index=False)
print(df.groupby("method")[[c for c in df.columns if c.startswith("rel_l2") or c == "near_wall_ratio"]].median().round(4).to_string())
