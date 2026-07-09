"""Canonical, mesh-agnostic case record shared by every data source and solver.

Conventions:
  * lengths in units of L_ref, the characteristic length declared by each source;
  * geometry-relative origin: (x, y) at the centroid of the solid geometry's footprint, z = 0 on the ground,
    so no dataset box or origin enters the features;
  * fields [Ux, Uy, Uz, Cp, k, eps]: U / U_ref, Cp = 2 p / U_ref^2 (kinematic p), k / U_ref^2,
    eps * L_ref / U_ref^3. eps exists only for k-epsilon-family closures; nu_t is never a shared target;
  * a channel group a source lacks is zero-filled and flagged 0 in `presence` (groups: U, Cp, k, eps);
  * `prior` is an optional low-fidelity solution on the same points, same channels, own presence flags.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

FIELDS = ["Ux", "Uy", "Uz", "Cp", "k", "eps"]
GROUPS = ["U", "Cp", "k", "eps"]
GROUP_CHANNELS = {"U": [0, 1, 2], "Cp": [3], "k": [4], "eps": [5]}
CHANNEL_GROUP = np.array([0, 0, 0, 1, 2, 3])
N_FIELDS = len(FIELDS)
CLOSURES = ["k-epsilon", "realizable-k-epsilon", "k-omega-SST", "spalart-allmaras",
            "hybrid-RANS-LES", "WMLES", "unknown"]
GROUNDS = ["stationary", "moving", "none", "unknown"]
PRIOR_KINDS = ["none", "coarse-twin", "potential-flow", "mapped-field", "other"]

# heavy-tailed turbulence channels are modelled as log1p(x / scale), then standardised per channel
K_SCALE = 1.0e-3
EPS_SCALE = 1.0e-3


@dataclass
class Conditioning:
    """Per-case boundary-condition and physics descriptors. `None` = unknown (presence flag 0)."""
    log10_re: Optional[float] = None
    closure: str = "unknown"
    abl_alpha: Optional[float] = None          # power-law exponent of the inflow profile
    log10_z0: Optional[float] = None           # log10(roughness length / L_ref)
    turb_intensity: Optional[float] = None
    ground: str = "unknown"
    inflow_dir: tuple = (1.0, 0.0, 0.0)        # free-stream direction; rotates with the geometry under augmentation
    has_ground: bool = True                    # height-above-ground is meaningful
    rotation_ok: bool = False                  # rotation about z is an exact symmetry of this set-up
    prior_kind: str = "none"

    def __post_init__(self):
        assert self.closure in CLOSURES, self.closure
        assert self.ground in GROUNDS, self.ground
        assert self.prior_kind in PRIOR_KINDS, self.prior_kind

    def scalar_vector(self) -> np.ndarray:
        """12-dim vector: 4 standardised scalars, their 4 presence flags, inflow direction (3), has_ground."""
        vals, pres = [], []
        for v, (mu, sd) in ((self.log10_re, (6.0, 2.0)), (self.abl_alpha, (0.2, 0.1)),
                            (self.log10_z0, (-3.0, 1.5)), (self.turb_intensity, (0.1, 0.05))):
            ok = v is not None and np.isfinite(v)
            vals.append((float(v) - mu) / sd if ok else 0.0); pres.append(float(ok))
        d = np.asarray(self.inflow_dir, np.float64); d = d / (np.linalg.norm(d) + 1e-12)
        return np.asarray(vals + pres + list(d) + [float(self.has_ground)], np.float32)


@dataclass
class Case:
    case_id: str
    source: str
    points: np.ndarray                      # (N,3) float32
    wall_dist: np.ndarray                   # (N,) distance to the nearest wall / L_ref
    normal: np.ndarray                      # (N,3) unit vector from the nearest wall point towards the point
    fields: Optional[np.ndarray]            # (N,6) or None for geometry-only (inference) cases
    presence: np.ndarray                    # (4,) bool per group U, Cp, k, eps
    cond: Conditioning
    L_ref: float
    U_ref: float
    cell_size: Optional[np.ndarray] = None  # (N,) local mesh spacing / L_ref
    prior: Optional[np.ndarray] = None      # (N,6) low-fidelity solution
    prior_presence: Optional[np.ndarray] = None
    stratum: Optional[np.ndarray] = None    # (N,) uint8: 0 uniform, 1 near-wall, 2 wake
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    def height(self) -> np.ndarray:
        return self.points[:, 2] if self.cond.has_ground else np.zeros(self.n, np.float32)

    def streamwise(self) -> np.ndarray:
        d = np.asarray(self.cond.inflow_dir, np.float32)
        return self.points @ (d / (np.linalg.norm(d) + 1e-12))

    # ---------------------------------------------------------------- fp16 shard I/O
    def save(self, path: str) -> None:
        cond = asdict(self.cond); cond["inflow_dir"] = [float(x) for x in self.cond.inflow_dir]
        payload = dict(points=self.points.astype(np.float32), wall_dist=self.wall_dist.astype(np.float16),
                       normal=self.normal.astype(np.float16), presence=self.presence.astype(np.uint8),
                       header=json.dumps(dict(case_id=self.case_id, source=self.source, L_ref=self.L_ref,
                                              U_ref=self.U_ref, cond=cond, meta=self.meta)))
        for k in ("fields", "cell_size", "prior"):
            v = getattr(self, k)
            if v is not None:
                payload[k] = v.astype(np.float16)
        if self.prior_presence is not None:
            payload["prior_presence"] = self.prior_presence.astype(np.uint8)
        if self.stratum is not None:
            payload["stratum"] = self.stratum.astype(np.uint8)
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Case":
        with np.load(path) as z:
            h = json.loads(str(z["header"]))
            c = h["cond"]; c["inflow_dir"] = tuple(c["inflow_dir"])
            g = lambda k, t=np.float32: z[k].astype(t) if k in z.files else None  # noqa: E731
            return cls(case_id=h["case_id"], source=h["source"], points=g("points"), wall_dist=g("wall_dist"),
                       normal=g("normal"), fields=g("fields"), presence=z["presence"].astype(bool),
                       cond=Conditioning(**c), L_ref=float(h["L_ref"]), U_ref=float(h["U_ref"]),
                       cell_size=g("cell_size"), prior=g("prior"), prior_presence=g("prior_presence", bool),
                       stratum=g("stratum", np.uint8), meta=h.get("meta", {}))


def encode_fields(f):
    """normalised physical fields -> network space (log1p on k and eps); numpy or torch."""
    try:
        import torch
        if isinstance(f, torch.Tensor):
            g = f.clone()
            g[..., 4] = torch.log1p(f[..., 4].clamp(min=0.0) / K_SCALE)
            g[..., 5] = torch.log1p(f[..., 5].clamp(min=0.0) / EPS_SCALE)
            return g
    except ImportError:  # pragma: no cover
        pass
    g = np.array(f, dtype=np.float32, copy=True)
    g[..., 4] = np.log1p(np.clip(g[..., 4], 0.0, None) / K_SCALE)
    g[..., 5] = np.log1p(np.clip(g[..., 5], 0.0, None) / EPS_SCALE)
    return g


def decode_fields(g):
    """inverse of encode_fields; numpy or torch."""
    try:
        import torch
        if isinstance(g, torch.Tensor):
            f = g.clone()
            f[..., 4] = torch.expm1(g[..., 4].clamp(min=0.0)) * K_SCALE
            f[..., 5] = torch.expm1(g[..., 5].clamp(min=0.0)) * EPS_SCALE
            return f
    except ImportError:  # pragma: no cover
        pass
    f = np.array(g, dtype=np.float32, copy=True)
    f[..., 4] = np.expm1(np.clip(f[..., 4], 0.0, None)) * K_SCALE
    f[..., 5] = np.expm1(np.clip(f[..., 5], 0.0, None)) * EPS_SCALE
    return f
