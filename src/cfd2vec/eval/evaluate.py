"""Full-field evaluation on held-out cases, with reference baselines."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..schema import Case
from ..tasks.predict import predict_case
from .metrics import near_wall_ratio, rel_l2


def evaluate_cases(model, paths, device="cpu", modes=("geometry-only", "with-prior"), baselines=True, seed=0):
    rows = []
    for p in paths:
        c = Case.load(p)
        base = dict(case_id=c.case_id, n_points=c.n, tier=c.meta.get("tier"), morphology=c.meta.get("morphology"))
        for mode in modes:
            use_prior = mode == "with-prior"
            if use_prior and c.prior is None:
                continue
            out = predict_case(model, c, use_prior=use_prior, device=device, seed=seed)
            r = dict(base, method=f"cfd2vec:{mode}")
            r.update(rel_l2(out["fields"], c.fields, c.presence)); r.update(near_wall_ratio(out["fields"], c.fields, c.wall_dist))
            rows.append(r)
        if baselines:
            if c.prior is not None:
                r = dict(base, method="prior alone (coarse twin)")
                r.update(rel_l2(c.prior, c.fields, c.presence)); r.update(near_wall_ratio(c.prior, c.fields, c.wall_dist))
                rows.append(r)
            mean = np.zeros_like(c.fields)
            r = dict(base, method="uniform inflow (U = U_ref e_x, Cp = k = eps = 0)")
            mean[:, 0:3] = np.asarray(c.cond.inflow_dir, np.float32)
            r.update(rel_l2(mean, c.fields, c.presence)); rows.append(r)
    return pd.DataFrame(rows)


def evaluate_cli(a):
    from ..api import CFD2vec
    m = CFD2vec.from_pretrained(a.ckpt, device=a.device)
    df = evaluate_cases(m.net, a.cases, m.device, modes=("geometry-only", "with-prior") if a.use_prior else ("geometry-only",))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby("method")[[c for c in df.columns if c.startswith("rel_l2")]].median().to_string())
