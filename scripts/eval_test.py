"""Held-out test evaluation of one or more checkpoints on identical cells.

    python scripts/eval_test.py --model S=runs/S_urban_v2_masked/last.pt@/path/shards_v2 \
                                --model pilot=runs/pilot_masked/last.pt@/path/shards_v1 --points 32768 --out results/test_S_vs_pilot.csv

Each case is scored on a fixed uniform sample of `--points` cells (0 = every cell), drawn with the same seed for every
model, so all models and baselines see identical cells. Modes: geometry only (fully masked) and with the coarse prior.
Baselines: coarse prior alone, uniform inflow.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from cfd2vec.api import CFD2vec
from cfd2vec.data.sampling import subsample
from cfd2vec.eval.metrics import near_wall_ratio, rel_l2
from cfd2vec.schema import Case
from cfd2vec.tasks.predict import predict_case
from cfd2vec.train.dataset import shard_paths

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help="name=checkpoint@shard_root")
ap.add_argument("--split", default="test"); ap.add_argument("--points", type=int, default=32768)
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
specs = []
for m in a.model:
    name, rest = m.split("=", 1); ck, root = rest.split("@", 1)
    specs.append((name, CFD2vec.from_pretrained(ck, device=a.device), root))
ref_root = specs[0][2]
paths = shard_paths(os.path.join(ref_root, "manifest_aero_urban.parquet"), a.split)
if a.limit:
    paths = paths[: a.limit]
rows, t0 = [], time.time()
for i, p in enumerate(paths):
    rel = os.path.relpath(p, ref_root)
    base_case = Case.load(p)
    rng = np.random.default_rng(1234 + i)
    idx = np.sort(rng.choice(base_case.n, min(a.points, base_case.n), replace=False)) if a.points else np.arange(base_case.n)
    info = dict(case_id=base_case.case_id, tier=base_case.meta.get("tier"), morphology=base_case.meta.get("morphology"),
                n_cells=base_case.n, n_scored=len(idx))
    q0 = subsample(base_case, idx)
    for name, model, root in specs:
        c = Case.load(os.path.join(root, rel))
        assert c.n == base_case.n and np.allclose(c.points[idx[:10]], base_case.points[idx[:10]]), "shard sets differ"
        q = subsample(c, idx)
        for mode in ("geometry-only", "with-prior"):
            out = predict_case(model.net, c, query=q, use_prior=mode == "with-prior", device=a.device, seed=0)
            r = dict(info, model=name, method=f"{name}:{mode}")
            r.update(rel_l2(out["fields"], q.fields, c.presence)); r.update(near_wall_ratio(out["fields"], q.fields, q.wall_dist))
            rows.append(r)
    r = dict(info, model="baseline", method="coarse prior alone")
    r.update(rel_l2(q0.prior, q0.fields, base_case.presence)); r.update(near_wall_ratio(q0.prior, q0.fields, q0.wall_dist))
    rows.append(r)
    uni = np.zeros_like(q0.fields); uni[:, 0:3] = np.asarray(base_case.cond.inflow_dir, np.float32)
    r = dict(info, model="baseline", method="uniform inflow"); r.update(rel_l2(uni, q0.fields, base_case.presence)); rows.append(r)
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{len(paths)} cases, {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
cols = [c for c in df.columns if c.startswith("rel_l2") or c == "near_wall_ratio"]
print(df.groupby("method")[cols].median().round(4).to_string())
