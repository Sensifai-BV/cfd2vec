"""Write Re into existing shards: copy <src> to <dst> with cond.log10_re = log10(U_ref H / nu) and meta.nu set.
The source shards are not modified, so a run reading them stays consistent.

    python scripts/backfill_re.py --src <shards> --dst <shards_v2> --params <params_extract.json>

params_extract.json: {"<tier>/<case>": {"nu": ..., "Uref": ..., "alpha": ...}} from each case's params.json.
Uref and alpha are cross-checked against the shard header; any mismatch aborts that case.
"""
import argparse
import json
import os
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cfd2vec.schema import Case


def one(job):
    src, dst, rec = job
    c = Case.load(src)
    key = c.case_id.replace("aero_urban/", "")
    if rec is None:
        return dict(case_id=c.case_id, error="no params")
    if abs(rec["Uref"] - c.U_ref) > 1e-9 or abs(rec["alpha"] - c.cond.abl_alpha) > 1e-9:
        return dict(case_id=c.case_id, error=f"Uref/alpha mismatch for {key}")
    c.cond.log10_re = float(np.log10(c.U_ref * c.L_ref / rec["nu"]))
    c.meta["nu"] = rec["nu"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    c.save(dst)
    return dict(case_id=c.case_id, log10_re=c.cond.log10_re, nu=rec["nu"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--params", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    P = json.load(open(a.params))
    man = pd.read_parquet(os.path.join(a.src, "manifest_aero_urban.parquet"))
    jobs = []
    for s in man.shard:
        rel = os.path.relpath(s, a.src)
        key = "/".join(rel.split(os.sep)[1:]).replace(".npz", "")
        jobs.append((s, os.path.join(a.dst, rel), P.get(key)))
    with Pool(a.workers) as p:
        res = pd.DataFrame(p.map(one, jobs, chunksize=8))
    bad = res["error"].notna().sum() if "error" in res else 0
    if bad:
        print(res[res["error"].notna()].head()); raise SystemExit(f"{bad} cases failed; manifest not written")
    out = man.drop(columns=["log10_re"], errors="ignore").merge(res[["case_id", "log10_re", "nu"]], on="case_id")
    out["shard"] = [os.path.join(a.dst, os.path.relpath(s, a.src)) for s in out.shard]
    out.to_parquet(os.path.join(a.dst, "manifest_aero_urban.parquet"), index=False)
    shutil.copy(a.params, os.path.join(a.dst, "source_params_aero_urban.json"))
    print(f"wrote {len(out)} shards to {a.dst}; log10 Re {out.log10_re.min():.3f}..{out.log10_re.max():.3f}")
