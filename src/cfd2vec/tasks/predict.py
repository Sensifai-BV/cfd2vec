"""Field prediction on an arbitrary point set."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..schema import Case, decode_fields
from ..train.dataset import SampleSpec, make_sample, to_torch


def spec_from_config(cfg, **kw) -> SampleSpec:
    return SampleSpec(n_context=cfg.n_context, n_query=cfg.n_query, n_tokens_scale1=cfg.n_tokens_scale1,
                      n_tokens_scale2=cfg.n_tokens_scale2, k_neighbors=cfg.k_neighbors, r1=cfg.r1, r2=cfg.r2, **kw)


@torch.no_grad()
def predict_case(model, case: Case, query: Optional[Case] = None, use_prior: bool = False, seed: int = 0,
                 chunk: int = 16384, device: str = "cpu", n_ensemble: int = 1, return_embedding: bool = False,
                 strip_labels: bool = True):
    """Predict normalised fields [U, Cp, k, eps] at the points of `query` (default: all points of `case`).

    Only geometry, mesh, BC, physics and (optionally) the prior of `case` are used. With `strip_labels` (default)
    the stored target fields and target presence are removed before the sample is built, so evaluation sees exactly
    the input a field-free deployment case gives. `strip_labels=False` reproduces the pre-fix evaluation of
    schema-1 checkpoints, whose fully masked output depended on stored label presence.
    `n_ensemble` > 1 averages over independent context samples and returns their spread (`std`): a sensitivity of the
    prediction to the context sample, not a calibrated predictive uncertainty.
    """
    from ..data.sampling import strip_targets
    model.eval()
    query = case if query is None else query
    ctx = strip_targets(case) if strip_labels else case
    spec = spec_from_config(model.cfg, fixed_ratio=1.0, augment=False, use_prior=use_prior)
    preds, embs = [], []
    for e in range(n_ensemble):
        rng = np.random.default_rng(seed + e)
        s = to_torch(make_sample(ctx, spec, rng, query_case=query))
        b = {k: v[None].to(device) for k, v in s.items()}
        mem, pos, glob = model.encode(b)
        out = model.decode(mem, pos, b, chunk=chunk)
        preds.append(decode_fields(model.unstandardise(out))[0].float().cpu().numpy())
        embs.append(glob[0].float().cpu().numpy())
    P = np.stack(preds)
    if not np.isfinite(P).all():
        raise FloatingPointError(f"{case.case_id}: non-finite predicted fields after the inverse transform")
    res = dict(fields=P.mean(0), std=P.std(0) if n_ensemble > 1 else None)
    if return_embedding:
        res["embedding"] = np.mean(embs, 0)
    return res
