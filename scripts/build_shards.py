"""Convert a source into fp16 `Case` shards plus manifest.parquet.

Train cases keep a stratified subset (near-wall / wake / uniform); val and test cases keep every point so that
metrics are computed on the full field.
"""
import argparse
import json
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cfd2vec.data.sampling import stratified_indices, subsample
from cfd2vec.paths import expand
from cfd2vec.sources import aero_urban


def one(job):
    path, out, split, n_keep, seed, nu = job
    try:
        v = aero_urban.read_voxel_npz(path)
        case, vort = aero_urban.voxel_to_case(v, split=split, return_vort=True, nu=nu)
        if split == "train":
            idx, strat = stratified_indices(case.wall_dist, vort, n_keep, np.random.default_rng(seed))
        else:
            idx, strat = stratified_indices(case.wall_dist, vort, case.n, np.random.default_rng(seed))
        case = subsample(case, idx); case.stratum = strat
        case.meta["sampling"] = "stratified" if split == "train" else "full"
        case.save(out)
        m = case.meta
        return dict(case_id=case.case_id, source=case.source, split=split, shard=out, n_points=case.n,
                    n_cells=m["n_cells"], tier=m["tier"], morphology=m["morphology"], L_ref=case.L_ref,
                    U_ref=case.U_ref, closure=case.cond.closure, fidelity=m["fidelity"], solver=m["solver"],
                    mesh=m["mesh"], abl_alpha=case.cond.abl_alpha, turb_intensity=case.cond.turb_intensity,
                    log10_re=case.cond.log10_re, prior=case.cond.prior_kind, converged=m["converged"],
                    diverged=m["diverged"], H=m["H"], Lz=m["Lz"])
    except Exception as e:  # noqa: BLE001
        return dict(case_id=path, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="${CFD2VEC_DATA}/aero_corpus_cache")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--out", default="${CFD2VEC_DATA}/cfd2vec_shards")
    ap.add_argument("--n-keep", type=int, default=65536)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--params", help="JSON {tier/case: {nu, Uref, ...}} extracted from each case's params.json")
    a = ap.parse_args(); a.cache, a.out = expand(a.cache), expand(a.out)
    S = json.load(open(a.splits))["sources"]["aero_urban"]
    P = json.load(open(a.params)) if a.params else {}
    jobs = []
    for tier in ("M", "L", "XL"):
        os.makedirs(os.path.join(a.out, "aero_urban", tier), exist_ok=True)
        for split in ("train", "val", "test"):
            for i, c in enumerate(S[tier].get(split, [])):
                out = os.path.join(a.out, "aero_urban", tier, f"{c}.npz")
                nu = P.get(f"{tier}/{c}", {}).get("nu")
                jobs.append((os.path.join(a.cache, tier, f"{c}.npz"), out, split, a.n_keep, zlib.crc32(f"{tier}/{c}".encode()), nu))
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time(); rows = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(one, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(a.out, "manifest_aero_urban.parquet"), index=False)
    err = df["error"].notna().sum() if "error" in df else 0
    print(f"done {len(df)} cases, {err} errors, {time.time() - t0:.0f}s")
