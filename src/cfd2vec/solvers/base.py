"""Common solver-adapter interface and geometry helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..schema import Case, Conditioning


@dataclass
class RunReport:
    iterations: int
    converged: bool
    exec_time_s: float
    residuals: dict = field(default_factory=dict)   # field -> list of initial residuals per iteration
    diverged: bool = False
    log_path: str = ""
    residual_iters: dict = field(default_factory=dict)   # field -> solver iteration (time) id of each residual
    returncode: Optional[int] = None                # solver exit status; None when not run by the adapter

    @property
    def failed(self) -> bool:
        return self.returncode not in (None, 0)


class SolverAdapter(ABC):
    name = "base"

    @abstractmethod
    def read_case(self, case_dir: str, U_ref: float, L_ref: float, cond: Conditioning, **kw) -> Case: ...

    @abstractmethod
    def write_initial(self, case_dir: str, fields: dict, **kw) -> list: ...

    def run(self, case_dir: str, command: Sequence[str], **kw) -> RunReport:
        raise NotImplementedError(f"{self.name}: running the solver is not supported by this adapter")


def wall_geometry(points: np.ndarray, face_centres: np.ndarray, face_normals: np.ndarray):
    """Distance from each point to the nearest wall-face centre, and the unit vector from that face towards the
    point (falls back to the face normal when the point sits on the face). Face-centre distance approximates the
    true point-to-face distance to within half a face size."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(face_centres).query(points, workers=-1)
    v = points - face_centres[j]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    nrm = np.where(n > 1e-12, v / np.maximum(n, 1e-12), face_normals[j])
    return d.astype(np.float32), nrm.astype(np.float32)


def orient_normals_to_fluid(face_centres: np.ndarray, face_normals: np.ndarray,
                            owner_centres: np.ndarray | None = None):
    """Orient wall-face normals from the wall into the fluid with verified face-owner connectivity: a normal that
    points away from the centre of the fluid cell that owns its face is flipped. Without `owner_centres` the
    normals are returned exactly as supplied and the report says so (`verified: False`). No geometric heuristic is
    applied: proximity does not identify the adjacent cell (a cell across a thin solid can be nearer than a
    stretched adjacent cell, and cube-root cell size bounds no extent of an elongated cell). Returns (normals,
    report)."""
    fn = np.asarray(face_normals, np.float64).copy()
    if owner_centres is None:
        return fn, dict(method="reader", verified=False, n_faces=int(len(fn)), flipped=0.0,
                        note="no verified face-owner connectivity; orientation as supplied by the reader")
    fc, oc = np.asarray(face_centres, np.float64), np.asarray(owner_centres, np.float64)
    if oc.shape != fc.shape or fn.shape != fc.shape:
        raise ValueError(f"owner_centres {oc.shape} and normals {fn.shape} must match face_centres {fc.shape}")
    flip = np.einsum("ij,ij->i", fn, oc - fc) < 0
    fn[flip] *= -1.0
    return fn, dict(method="owner", verified=True, n_faces=int(len(fn)), flipped=float(flip.mean()) if len(fn) else 0.0)


def footprint_stats(face_centres: np.ndarray, face_normals: np.ndarray, face_areas: np.ndarray,
                    ground_mask: Optional[np.ndarray], ground_z: Optional[float]):
    """Plan-view footprint of the solid geometry from its upward-facing wall faces (projected area n_z * A).
    Returns (origin, mean_height): origin = footprint centroid on the ground plane; mean_height = projected-area
    weighted mean height of the upward faces above the ground (the mean building height for urban layouts)."""
    body = np.ones(len(face_centres), bool) if ground_mask is None else ~ground_mask
    if not body.any():
        body = np.ones(len(face_centres), bool)
    w = np.clip(face_normals[body, 2], 0, None) * face_areas[body]      # normals point from the wall into the fluid
    c = face_centres[body]
    if w.sum() <= 0:
        w = face_areas[body]
    z0 = ground_z if ground_z is not None else float(c[:, 2].min())
    origin = np.array([np.average(c[:, 0], weights=w), np.average(c[:, 1], weights=w), z0], np.float64)
    return origin, float(np.average(c[:, 2] - z0, weights=w))


def eps_mixing_length(k: np.ndarray, wall_dist: np.ndarray, L_ref: float, c_mu: float = 0.09, kappa: float = 0.41,
                      l_max_frac: float = 0.1) -> np.ndarray:
    """epsilon = C_mu^(3/4) k^(3/2) / l with l = min(kappa d, l_max_frac L_ref); used when no eps prediction exists."""
    l = np.minimum(kappa * np.maximum(wall_dist, 1e-6), l_max_frac * L_ref)
    return c_mu ** 0.75 * np.clip(k, 0, None) ** 1.5 / l
