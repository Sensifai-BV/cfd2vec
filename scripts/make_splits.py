"""Freeze configs/splits.json for the AERO urban source.

Geometry-separated by case id. Existing warm-start splits are reused where they exist so that results stay
comparable with the earlier warm-start model; tiers without one get a seeded split. Diverged cases are excluded.
"""
import argparse
import json
import os

import numpy as np

from cfd2vec.paths import expand

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="${CFD2VEC_DATA}/aero_corpus_cache")
ap.add_argument("--reuse", nargs="*", default=[
    "M=~/External/Projects/aerofoam/ml/runs/M_base24_fp32/split_used.json",
    "L=~/External/Projects/aerofoam/ml/runs/L_ft24/split_used.json"])
ap.add_argument("--bench", default="${CFD2VEC_DATA}/aero_bench")
ap.add_argument("--out", default="configs/splits.json")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); a.cache, a.bench = expand(a.cache), expand(a.bench)

reuse = {k: expand(v) for k, v in (r.split("=", 1) for r in a.reuse)}
out = {"version": 1, "frozen": "2026-09-28", "sources": {"aero_urban": {}}}
for tier in ("M", "L", "XL"):
    idx = json.load(open(os.path.join(a.cache, f"index_{tier}.json")))
    ok = sorted(r["case"] for r in idx if "error" not in r and not r.get("diverged") and r.get("converged_fine", True))
    if tier in reuse and os.path.exists(reuse[tier]):
        S = json.load(open(reuse[tier]))
        S = {k: sorted(set(v) & set(ok)) for k, v in S.items()}
        origin = os.path.relpath(reuse[tier], expand("~/External/Projects"))
    else:
        rng = np.random.default_rng(a.seed); perm = list(rng.permutation(ok))
        S = {"train": sorted(perm[1:]), "val": sorted(perm[:1]), "test": []}
        origin = f"seeded (seed={a.seed}); held-out XL-scale tests are the bench cases"
    out["sources"]["aero_urban"][tier] = dict(S, origin=origin, n_valid=len(ok))
bench = sorted(d for d in os.listdir(a.bench) if d.startswith("B") and os.path.isdir(os.path.join(a.bench, d)))
out["sources"]["aero_urban"]["bench"] = {"test": bench, "origin": "solver warm-start benchmark; never trained on"}
json.dump(out, open(a.out, "w"), indent=1)
for t, S in out["sources"]["aero_urban"].items():
    print(t, {k: len(v) for k, v in S.items() if isinstance(v, list)})
