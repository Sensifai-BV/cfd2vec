"""Masked field modelling: per-sample mask ratio and structured (block / region) masks.

Geometry, mesh, boundary conditions and physics always stay visible; only field values are hidden.
Every downstream use corresponds to the fully masked mode (ratio 1.0), so it is drawn often.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

MODES = ("tokens", "near_wall", "wake", "boxes")


def draw_ratio(rng: np.random.Generator, ratios=(0.6, 0.9, 1.0), probs=(0.5, 0.25, 0.25)) -> float:
    return float(rng.choice(ratios, p=probs))


def point_mask(points: np.ndarray, wall_dist: np.ndarray, streamwise: np.ndarray, centres: np.ndarray,
               ratio: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Boolean (N,) - True where the field is hidden. The top `ratio` fraction of a mode-specific score is masked:
       tokens     whole token regions (score shared by all points nearest to a coarse centre)
       near_wall  the band closest to walls
       wake       the downstream band
       boxes      unions of random axis-aligned blocks
    """
    N = points.shape[0]
    if ratio >= 1.0:
        return np.ones(N, bool)
    if mode == "tokens":
        _, owner = cKDTree(points[centres]).query(points)
        score = rng.random(len(centres))[owner]
    elif mode == "near_wall":
        score = -wall_dist + 1e-3 * rng.standard_normal(N)
    elif mode == "wake":
        score = streamwise + 1e-3 * rng.standard_normal(N)
    elif mode == "boxes":
        seeds = points[rng.integers(0, N, size=8)]
        half = rng.uniform(0.3, 1.5, size=(8, 3))
        cheb = np.max(np.abs(points[:, None, :] - seeds[None]) / half[None], axis=-1)
        score = -cheb.min(axis=1)
    else:
        raise ValueError(mode)
    thr = np.quantile(score, 1.0 - ratio)
    return score >= thr
