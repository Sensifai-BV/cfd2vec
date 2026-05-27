"""Field metrics in normalised physical space (U / U_ref, Cp, k / U_ref^2, eps L_ref / U_ref^3)."""
from __future__ import annotations

import numpy as np

from ..schema import GROUP_CHANNELS, GROUPS


def rel_l2(pred: np.ndarray, true: np.ndarray, presence=None, weights=None) -> dict:
    """Relative L2 error per channel group; U is treated as one vector field."""
    out = {}
    w = np.ones(len(true)) if weights is None else weights
    for gi, g in enumerate(GROUPS):
        if presence is not None and not presence[gi]:
            continue
        ch = GROUP_CHANNELS[g]
        num = np.sqrt((w[:, None] * (pred[:, ch] - true[:, ch]) ** 2).sum())
        den = np.sqrt((w[:, None] * true[:, ch] ** 2).sum()) + 1e-12
        out[f"rel_l2_{g}"] = float(num / den)
    return out


def near_wall_ratio(pred, true, wall_dist, near: float = 0.25, far: float = 1.0) -> dict:
    """RMS velocity error (units of U_ref) near walls vs in the free stream, and their ratio.
    Recommended acceptance for a tokeniser: ratio below 2."""
    e = np.linalg.norm(pred[:, 0:3] - true[:, 0:3], axis=1)
    a, b = wall_dist < near, wall_dist > far
    rn = float(np.sqrt((e[a] ** 2).mean())) if a.any() else float("nan")
    rf = float(np.sqrt((e[b] ** 2).mean())) if b.any() else float("nan")
    return dict(rmse_U_near_wall=rn, rmse_U_free_stream=rf, near_wall_ratio=rn / rf if rf > 0 else float("nan"))


def summarise(rows: list) -> dict:
    keys = sorted({k for r in rows for k in r if isinstance(r[k], float)})
    return {k: float(np.nanmedian([r[k] for r in rows if k in r])) for k in keys}
