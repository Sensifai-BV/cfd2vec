"""Point sampling and exact symmetry augmentations for `Case` records."""
from __future__ import annotations

import copy

import numpy as np

from ..schema import Case

_PER_POINT = ("points", "wall_dist", "normal", "fields", "cell_size", "prior", "stratum")


def stratified_indices(wall_dist: np.ndarray, wake_indicator: np.ndarray, n_keep: int, rng: np.random.Generator,
                       near_wall_band: float = 0.25, frac=(0.5, 0.3, 0.2)):
    """Near-wall / wake / uniform stratified subsample (default 50 / 30 / 20 %).

    Draw order is near-wall (wall_dist < near_wall_band, L_ref units), then uniform over everything not yet taken,
    then wake (top decile of `wake_indicator`, any scalar that peaks in the wake) from what remains. Strata 1 and 0
    therefore depend on geometry only and may serve as model context; stratum 2 depends on the solution and must
    only be used as loss queries, otherwise point density would reveal the flow. A short pool is not refilled.
    When every point is kept (n <= n_keep) there is nothing to stratify: strata are near-wall / uniform by geometry
    alone and no point is withheld from the context, so context membership never depends on the solution.
    Returns (sorted idx, stratum) with stratum 1 near-wall, 0 uniform, 2 wake.
    """
    n = wall_dist.shape[0]
    if n <= n_keep:
        s = np.zeros(n, np.uint8)
        s[wall_dist < near_wall_band] = 1
        return np.arange(n), s
    thr = np.quantile(wake_indicator, 0.9)
    taken = np.zeros(n, bool)
    out, strata = [], []

    def draw(pool, q, code):
        m = min(int(round(q * n_keep)), pool.size)
        pick = rng.choice(pool, size=m, replace=False)
        taken[pick] = True; out.append(pick); strata.append(np.full(m, code, np.uint8))

    draw(np.flatnonzero(wall_dist < near_wall_band), frac[0], 1)
    draw(np.flatnonzero(~taken), frac[2], 0)
    draw(np.flatnonzero((wake_indicator >= thr) & ~taken), frac[1], 2)
    idx = np.concatenate(out); s = np.concatenate(strata)
    o = np.argsort(idx)
    return idx[o], s[o]


def context_indices(wall_dist: np.ndarray, n: int, rng: np.random.Generator, near_wall_band: float = 0.25,
                    near_wall_frac: float = 5.0 / 7.0) -> np.ndarray:
    """Geometry-only context sample with the same near-wall share as the stratified training shards.
    Used for full-field cases (validation, test, inference on a new mesh)."""
    N = wall_dist.shape[0]
    if N <= n:
        return np.arange(N)
    near = np.flatnonzero(wall_dist < near_wall_band)
    m = min(int(round(near_wall_frac * n)), near.size)
    a = rng.choice(near, size=m, replace=False)
    mask = np.ones(N, bool); mask[a] = False
    b = rng.choice(np.flatnonzero(mask), size=n - m, replace=False)
    return np.sort(np.concatenate([a, b]))


def strip_targets(case: Case) -> Case:
    """The same case as a deployment input: geometry, mesh, conditioning and prior kept; target fields and target
    presence removed. Metrics are then computed against the original case, held separately."""
    c = copy.copy(case)
    c.fields = None
    c.presence = np.zeros(4, bool)
    return c


def subsample(case: Case, idx: np.ndarray) -> Case:
    c = copy.copy(case)
    for k in _PER_POINT:
        v = getattr(case, k)
        setattr(c, k, None if v is None else v[idx])
    return c


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)


def _apply_linear(case: Case, M: np.ndarray) -> Case:
    c = copy.copy(case)
    c.points = case.points @ M.T
    c.normal = case.normal @ M.T
    for k in ("fields", "prior"):
        v = getattr(case, k)
        if v is not None:
            v = v.copy(); v[:, 0:3] = v[:, 0:3] @ M.T; setattr(c, k, v)
    cond = copy.copy(case.cond)
    cond.inflow_dir = tuple((M @ np.asarray(case.cond.inflow_dir, np.float32)).tolist())
    c.cond = cond
    return c


def rotation_z(theta: float) -> np.ndarray:
    return _rot(theta)


MIRROR_Y = np.diag([1.0, -1.0, 1.0]).astype(np.float32)


def transform_case(case: Case, M: np.ndarray) -> Case:
    """Apply an orthogonal map about the origin to geometry, flow, prior and inflow direction."""
    return _apply_linear(case, np.asarray(M, np.float32))


def rotate_case_z(case: Case, theta: float) -> Case:
    """Rotate geometry, flow, prior and inflow direction together about the vertical axis through the origin.

    For a uniform-direction inflow over flat ground this maps a solution onto the exact solution of the rotated
    problem. Scalars (Cp, k, eps, wall distance, height, cell size) are invariant.
    """
    return _apply_linear(case, _rot(theta))


def mirror_case_y(case: Case) -> Case:
    """Reflection y -> -y: the mirrored problem has the mirrored solution for any geometry."""
    return _apply_linear(case, np.diag([1.0, -1.0, 1.0]).astype(np.float32))
