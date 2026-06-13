"""Tokeniser geometry: two-scale farthest-point sampling and k-nearest-neighbour groups.

Scale index 0 (`scale == 0`, budget `n_tokens_scale1`) is wall-weighted (dense near walls, nominal scale r1); index 1
(budget `n_tokens_scale2`) is uniform over the domain (nominal scale r2).
Both are centre distributions with k-nearest-neighbour support, not bounded neighbourhoods.
Runs on CPU in data-loader workers for small point sets, or on the accelerator for large ones (same code).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


@torch.no_grad()
def farthest_point_sampling(points: torch.Tensor, n: int, weights: torch.Tensor | None = None,
                            seed: int = 0) -> torch.Tensor:
    """points (N,3) -> indices (n,). With `weights`, the next centre maximises weight * distance-to-selected-set,
    which concentrates centres where weights are large while still covering the whole set."""
    N = points.shape[0]
    if N == 0 or n < 1:
        raise ValueError(f"farthest-point sampling needs points and a positive budget (N={N}, n={n})")
    n = min(n, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.empty(n, dtype=torch.long, device=points.device)
    idx[0] = int(torch.randint(N, (1,), generator=g))
    mind = torch.full((N,), float("inf"), device=points.device)
    w = weights if weights is not None else torch.ones(N, device=points.device)
    for i in range(1, n):
        d = ((points - points[idx[i - 1]]) ** 2).sum(-1)
        mind = torch.minimum(mind, d)
        mind[idx[i - 1]] = -1.0          # never reselect: centres are unique indices; distinct coordinates win over
        idx[i] = torch.argmax(mind * w)  # duplicates of chosen ones because their score stays positive
    return idx


def build_tokens(points: np.ndarray, wall_dist: np.ndarray, n1: int, n2: int, k: int, r1: float, r2: float,
                 seed: int = 0) -> dict:
    """Token centres, neighbour groups and per-token scale descriptors for one point set (L_ref units).

    Both scales group each centre with its k nearest points; r1 and r2 are nominal coordinate scales (relative
    offsets are divided by them), not radius cut-offs, and the measured k-th-neighbour distance is returned as
    r_eff. Centre selection starts from a seeded index, so reordering the input points changes the tokens."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"points must be a non-empty (N, 3) array, got {points.shape}")
    if n1 < 1 or n2 < 1 or k < 1:
        raise ValueError(f"token budgets and neighbour count must be positive (n1={n1}, n2={n2}, k={k})")
    if not (r1 > 0 and r2 > 0 and np.isfinite(r1) and np.isfinite(r2)):
        raise ValueError(f"nominal radii must be finite and positive (r1={r1}, r2={r2})")
    if not (np.isfinite(points).all() and np.isfinite(wall_dist).all()):
        raise ValueError("points and wall distances must be finite")
    pts = torch.from_numpy(np.ascontiguousarray(points, np.float32))
    w = torch.from_numpy((1.0 / (1.0 + np.clip(np.asarray(wall_dist, np.float32), 0, None) / r1)) ** 2)
    c1 = farthest_point_sampling(pts, n1, w, seed)
    c2 = farthest_point_sampling(pts, n2, None, seed + 1)
    centres = torch.cat([c1, c2]).numpy()
    tree = cKDTree(points)
    kk = min(k, points.shape[0])
    dist, nb = tree.query(points[centres], kk)
    if kk == 1:
        dist, nb = dist[:, None], nb[:, None]
    scale = np.concatenate([np.zeros(len(c1), np.int64), np.ones(len(c2), np.int64)])
    radius = np.where(scale == 0, r1, r2).astype(np.float32)
    return dict(centre_idx=centres.astype(np.int64), nb_idx=nb.astype(np.int64), scale=scale,
                radius=radius, r_eff=dist[:, -1].astype(np.float32))


def coverage(n_points: int, nb_idx: np.ndarray) -> float:
    """Fraction of points that belong to at least one token group (points outside every group are invisible)."""
    seen = np.zeros(n_points, bool); seen[nb_idx.ravel()] = True
    return float(seen.mean())
