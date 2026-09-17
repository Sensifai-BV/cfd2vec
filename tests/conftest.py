import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cfd2vec.schema import Case, Conditioning  # noqa: E402


def synthetic_case(n=3000, seed=0, prior=True):
    """A single cube on the ground in a uniform stream; analytic-looking fields, enough for shape and symmetry tests."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-4, -3, 0], [8, 3, 3], size=(n * 2, 3)).astype(np.float32)
    inside = (np.abs(pts[:, 0]) < 0.5) & (np.abs(pts[:, 1]) < 0.5) & (pts[:, 2] < 1.0)
    pts = pts[~inside][:n]
    dx = np.maximum(np.abs(pts[:, 0]) - 0.5, 0); dy = np.maximum(np.abs(pts[:, 1]) - 0.5, 0)
    dz = np.maximum(pts[:, 2] - 1.0, 0)
    d_box = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    wd = np.minimum(d_box, pts[:, 2]).astype(np.float32)
    nrm = np.zeros_like(pts); nrm[:, 2] = 1.0
    U = np.stack([np.tanh(pts[:, 2]) * (1 - 0.5 * np.exp(-d_box)), 0.1 * np.sin(pts[:, 1]), 0.05 * np.cos(pts[:, 0])], 1)
    f = np.concatenate([U, (1 - (U ** 2).sum(1))[:, None], 0.01 * np.exp(-wd)[:, None], 0.02 * np.exp(-wd)[:, None]], 1)
    cond = Conditioning(closure="k-epsilon", abl_alpha=0.2, turb_intensity=0.1, ground="stationary",
                        inflow_dir=(1.0, 0.0, 0.0), rotation_ok=True, prior_kind="coarse-twin" if prior else "none")
    return Case(case_id=f"synthetic/{seed}", source="synthetic", points=pts, wall_dist=wd, normal=nrm,
                fields=f.astype(np.float32), presence=np.ones(4, bool), cond=cond, L_ref=1.0, U_ref=1.0,
                cell_size=np.full(len(pts), 0.05, np.float32),
                prior=(f * 0.9).astype(np.float32) if prior else None,
                prior_presence=np.ones(4, bool) if prior else None)


@pytest.fixture
def case():
    return synthetic_case()
