"""Encoder-usefulness diagnostics: does the prediction use the latent memory, the prior, and does the embedding carry
case-level information? Interventions (zeroed or swapped memory) are out of distribution, so a large degradation
supports memory use but its size is not an accuracy estimate.

    memory_dependence   normal memory vs zeroed spatial tokens vs spatial tokens of another case
    prior_dependence    geometry only, prior alone, model with prior
    gradient_transport  gradient norm per module group on one training batch
    frozen_readout      nested leave-one-out ridge probe from embeddings (trained / random encoder) or geometry
                        features; preprocessing and penalty are fitted inside each training fold
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..data.sampling import strip_targets, subsample
from ..schema import Case, decode_fields
from ..tasks.predict import spec_from_config
from ..train.dataset import make_sample, to_torch
from .metrics import rel_l2

GROUPS = {"pointnet": ("pointnet", "tok_proj", "tok_geom", "scale_emb"),
          "conditioning": ("cond_mlp", "closure_emb", "ground_emb", "prior_emb", "cls"),
          "encoder_first": ("blocks.0.",), "encoder_last": None, "decoder": ("q_mlp", "dec.", "dec_norm"),
          "head": ("head",)}


def _batch(model, case: Case, query: Case, use_prior: bool, seed: int, device: str) -> dict:
    spec = spec_from_config(model.cfg, fixed_ratio=1.0, augment=False, use_prior=use_prior)
    s = to_torch(make_sample(strip_targets(case), spec, np.random.default_rng(seed), query_case=query))
    return {k: v[None].to(device) for k, v in s.items()}


@torch.no_grad()
def _fields(model, mem, pos, b, chunk=8192):
    return decode_fields(model.unstandardise(model.decode(mem, pos, b, chunk=chunk)))[0].float().cpu().numpy()


@torch.no_grad()
def memory_dependence(model, cases: Sequence[Case], n_points: int = 8192, seed: int = 0, device: str = "cpu") -> list:
    """Per case: rel-L2 with its own memory, with the spatial tokens zeroed (conditioning tokens kept), and with the
    spatial tokens of the next case in the list (its token positions kept, so the rotary geometry is consistent)."""
    model.eval()
    rng = np.random.default_rng(seed)
    qs = [subsample(c, np.sort(rng.choice(c.n, min(n_points, c.n), replace=False))) for c in cases]
    enc = []
    for c, q in zip(cases, qs):
        b = _batch(model, c, q, False, seed, device)
        mem, pos, _ = model.encode(b)
        enc.append((b, mem, pos))
    nc = model.n_cond_tokens
    rows = []
    for i, (c, q) in enumerate(zip(cases, qs)):
        b, mem, pos = enc[i]
        _, mem_o, pos_o = enc[(i + 1) % len(cases)]
        zero = mem.clone(); zero[:, nc:] = 0.0
        swap = torch.cat([mem[:, :nc], mem_o[:, nc:]], 1)
        swap_pos = torch.cat([pos[:, :nc], pos_o[:, nc:]], 1)
        b_sw = dict(b); b_sw["tok_valid"] = enc[(i + 1) % len(cases)][0].get("tok_valid")
        r = dict(case_id=c.case_id)
        for name, m, p, bb in (("own", mem, pos, b), ("zeroed", zero, pos, b), ("swapped", swap, swap_pos, b_sw)):
            for k, v in rel_l2(_fields(model, m, p, bb), q.fields, c.presence).items():
                r[f"{name}_{k}"] = v
        rows.append(r)
    return rows


@torch.no_grad()
def prior_dependence(model, cases: Sequence[Case], n_points: int = 8192, seed: int = 0, device: str = "cpu") -> list:
    from ..tasks.predict import predict_case
    rng = np.random.default_rng(seed)
    rows = []
    for c in cases:
        q = subsample(c, np.sort(rng.choice(c.n, min(n_points, c.n), replace=False)))
        r = dict(case_id=c.case_id)
        for name, up in (("geometry_only", False), ("with_prior", True)):
            if up and c.prior is None:
                continue
            out = predict_case(model, c, query=q, use_prior=up, seed=seed, device=device)
            r.update({f"{name}_{k}": v for k, v in rel_l2(out["fields"], q.fields, c.presence).items()})
        if c.prior is not None:
            r.update({f"prior_alone_{k}": v for k, v in rel_l2(q.prior, q.fields, c.presence).items()})
        rows.append(r)
    return rows


def gradient_transport(model, batch: dict) -> dict:
    """Gradient norm per module group for one reconstruction loss on `batch` (train mode). With an exactly zero head
    only the head receives gradient on the first step; run it after at least one head update."""
    model.train(); model.zero_grad(set_to_none=True)
    loss, _ = model.loss(model(batch)[0], batch)
    loss.backward()
    last = f"blocks.{len(model.blocks) - 1}."
    out = {}
    for g, prefixes in GROUPS.items():
        pre = (last,) if prefixes is None else prefixes
        sq = [float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
              if p.grad is not None and any(n.startswith(x) for x in pre)]
        out[g] = float(np.sqrt(sum(sq))) if sq else 0.0
    out["loss"] = float(loss.detach())
    model.zero_grad(set_to_none=True); model.eval()
    return out


@torch.no_grad()
def embeddings(model, cases: Sequence[Case], seed: int = 0, device: str = "cpu") -> np.ndarray:
    """Global embeddings under a fixed contract: geometry only (no prior), seed `seed`, the model's context budget."""
    model.eval()
    E = []
    for c in cases:
        b = _batch(model, c, subsample(c, [0]), False, seed, device)
        E.append(model.encode(b)[2][0].float().cpu().numpy())
    return np.stack(E)


def case_targets(c: Case) -> dict:
    """Case-level flow quantities for a frozen probe (canonical units): mean speed at pedestrian height in the
    streets, mean near-wall Cp and domain-mean k."""
    z, wd = c.points[:, 2], c.wall_dist
    ped = (z > 0.05) & (z < 0.3) & (wd >= 0.9 * z)
    near = wd < 0.1
    sp = np.linalg.norm(c.fields[:, 0:3], axis=1)
    return dict(pedestrian_speed=float(sp[ped].mean()) if ped.any() else float("nan"),
                near_wall_cp=float(c.fields[near, 3].mean()) if near.any() else float("nan"),
                mean_k=float(c.fields[:, 4].mean()))


def geometry_features(c: Case) -> np.ndarray:
    """Simple case-level geometry descriptors: near-wall fraction, roof-height statistics, plan extent, point count."""
    wd, z = c.wall_dist, c.points[:, 2]
    near = wd < 0.1
    zz = z[near & (z > 0.05)]
    ext = np.ptp(c.points[:, :2], axis=0)
    return np.array([near.mean(), zz.mean() if zz.size else 0.0, zz.std() if zz.size else 0.0,
                     np.quantile(zz, 0.9) if zz.size else 0.0, ext[0], ext[1], np.log(c.n)], np.float64)


ALPHAS = tuple(10.0 ** np.arange(-3, 8))


def ridge_loo(X: np.ndarray, y: np.ndarray, alphas=ALPHAS) -> dict:
    """Single-level leave-one-out ridge: the same LOO errors choose the penalty and report performance, and features
    are standardised on all cases, so the R^2 is selection-biased (optimistic by an unquantified amount). Kept for
    comparison with `ridge_nested_loo`, which is the reported estimate."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    yc = y - y.mean()
    best = None
    for a in alphas:
        H = X @ np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T)
        res = (yc - H @ yc) / np.clip(1 - np.diag(H) - 1.0 / len(y), 1e-6, None)
        sse = float((res ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, a)
    return dict(r2=1 - best[0] / float((yc ** 2).sum()), alpha=best[1], n=int(len(y)),
                alpha_at_grid_edge=bool(best[1] in (min(alphas), max(alphas))))


def _dual_ridge_select(X: np.ndarray, y: np.ndarray, alphas, return_sse: bool = False):
    """Penalty with the lowest inner leave-one-out error on (X, y) only. Every inner fold is refitted with its own
    centring (features and target centred on the fold's training rows). Centring on the whole inner set instead
    would make each left-out row minus the sum of the others, so an interpolating fit predicts it exactly and the
    smallest penalty always wins when features outnumber cases. One eigendecomposition per fold serves all penalties:
    (K + a I)^-1 = V diag(1 / (s + a)) V^T."""
    n = len(y)
    alphas = np.asarray(alphas, float)
    sse = np.zeros(len(alphas))
    for i in range(n):
        m = np.arange(n) != i
        mu, ym = X[m].mean(0), y[m].mean()
        Xm = X[m] - mu
        s, V = np.linalg.eigh(Xm @ Xm.T)
        proj = V.T @ (y[m] - ym)                         # (n-1,)
        kv = V.T @ (Xm @ (X[i] - mu))                    # (n-1,)
        pred = ym + (kv[None, :] * proj[None, :] / (np.clip(s, 0, None)[None, :] + alphas[:, None])).sum(1)
        sse += (y[i] - pred) ** 2
    k = int(np.argmin(sse))
    return (float(sse[k]), float(alphas[k])) if return_sse else float(alphas[k])


def ridge_nested_loo(X: np.ndarray, y: np.ndarray, alphas=ALPHAS, n_boot: int = 2000, seed: int = 0) -> dict:
    """Nested leave-one-out ridge. For each held-out case: standardise features with the other cases' statistics,
    choose the penalty by inner leave-one-out on the other cases, fit on them, predict the held-out case. Nothing
    about the held-out case enters its own prediction. R^2 = 1 - SSE / SS_tot over the held-out predictions; the
    95 % interval resamples cases (bootstrap of the held-out errors, predictions fixed)."""
    ok = np.isfinite(y)
    X, y = np.asarray(X, float)[ok], np.asarray(y, float)[ok]
    n = len(y)
    pred, chosen = np.zeros(n), []
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xt, xi = (X[tr] - mu) / sd, (X[i] - mu) / sd
        a = _dual_ridge_select(Xt, y[tr], alphas); chosen.append(a)
        coef = np.linalg.solve(Xt @ Xt.T + a * np.eye(n - 1), y[tr] - y[tr].mean())
        pred[i] = y[tr].mean() + (Xt @ xi) @ coef
    err, dev = (y - pred) ** 2, (y - y.mean()) ** 2
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        d = ((y[k] - y[k].mean()) ** 2).sum()
        if d > 0:
            boots.append(1 - err[k].sum() / d)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return dict(r2=float(1 - err.sum() / dev.sum()), r2_ci95=[float(lo), float(hi)], n=int(n),
                alpha_median=float(np.median(chosen)),
                alpha_at_grid_edge_fraction=float(np.mean([a in (min(alphas), max(alphas)) for a in chosen])))


def frozen_readout(E_by_name: dict, cases: Sequence[Case]) -> list:
    """Nested-LOO R^2 (reported) and the selection-biased single-level LOO R^2 (for comparison) per feature set and
    case-level target."""
    T = [case_targets(c) for c in cases]
    rows = []
    for name, X in E_by_name.items():
        for t in T[0]:
            y = np.array([x[t] for x in T])
            nest = ridge_nested_loo(np.asarray(X, float), y)
            biased = ridge_loo(np.asarray(X, float), y)
            rows.append(dict(features=name, target=t, r2=nest["r2"], r2_ci95_lo=nest["r2_ci95"][0],
                             r2_ci95_hi=nest["r2_ci95"][1], n=nest["n"], alpha_median=nest["alpha_median"],
                             alpha_at_grid_edge_fraction=nest["alpha_at_grid_edge_fraction"],
                             r2_selection_biased=biased["r2"]))
    return rows


def random_like(model, seed: int = 0):
    """Same architecture and channel statistics, freshly initialised weights."""
    from ..model.network import CFD2vecNet
    torch.manual_seed(seed)
    return CFD2vecNet(model.cfg, model.ch_mean.tolist(), model.ch_std.tolist()).eval()


def label_leak_check(model, cases: Sequence[Case], n_points: int = 8192, seed: int = 0, device: str = "cpu") -> list:
    """Geometry-only rel-L2 with stored labels visible to the sample builder (historical evaluation of schema-1
    checkpoints) and with labels removed (deployment input)."""
    from ..tasks.predict import predict_case
    rng = np.random.default_rng(seed)
    rows = []
    for c in cases:
        q = subsample(c, np.sort(rng.choice(c.n, min(n_points, c.n), replace=False)))
        r = dict(case_id=c.case_id)
        for name, strip in (("labels_visible", False), ("field_free", True)):
            out = predict_case(model, c, query=q, use_prior=False, seed=seed, device=device, strip_labels=strip)
            r.update({f"{name}_{k}": v for k, v in rel_l2(out["fields"], q.fields, c.presence).items()})
        rows.append(r)
    return rows
