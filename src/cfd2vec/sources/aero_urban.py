"""AERO urban RANS corpus: steady k-epsilon solutions with a power-law ABL inflow along +x on Cartesian voxel
meshes, each with an exactly 2x coarser twin solved on the same problem.

Voxel cache per case (float16 npz):
    inp  (11, nx,ny,nz)  coarse twin nearest-upsampled: U/Uref (3), p/Uref^2, log1p(k/Uref^2), log1p(eps Lz/Uref^3),
                         fluid mask, wall distance/Lz (capped), z/Lz, alpha, I
    tgt  ( 6, nx,ny,nz)  fine solution: U/Uref (3), p/Uref^2, log1p(k/Uref^2), log1p(eps Lz/Uref^3)
    mask ( 1, ...)       fine fluid mask;  order (3, n_fluid) OpenFOAM cell -> voxel;  info (json)

The coarse twin enters the model as the `prior` field (super-fidelity / warm-start mode); geometry-only
prediction simply leaves it out.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from ..schema import Case, Conditioning

DX = 0.05
TIERS = ("M", "L", "XL")


def mean_building_height(solid: np.ndarray, dx: float = DX) -> float:
    """Area-weighted mean building height over footprint columns (solid at z = 0)."""
    foot = solid[:, :, 0]
    if not foot.any():
        return float("nan")
    nz = solid.shape[2]
    top = np.where(solid.all(axis=2), nz, np.argmin(solid, axis=2))
    return float(top[foot].mean() * dx)


def footprint_centroid(solid: np.ndarray, dx: float = DX) -> np.ndarray:
    ix, iy = np.nonzero(solid[:, :, 0])
    return np.array([(ix.mean() + 0.5) * dx, (iy.mean() + 0.5) * dx, 0.0], np.float32)


def wall_features(fluid: np.ndarray, dx: float = DX):
    """Euclidean distance from each voxel centre to the nearest wall face (buildings and the ground plane z = 0),
    and the unit vector from that wall towards the voxel. The nearest solid voxel centre lies dx/2 behind the face."""
    solid_g = np.pad(~fluid, ((0, 0), (0, 0), (1, 0)), constant_values=True)
    d, ind = ndimage.distance_transform_edt(~solid_g, sampling=dx, return_indices=True)
    d, ind = d[:, :, 1:], ind[:, :, :, 1:]
    vec = (np.indices(fluid.shape) + np.array([0, 0, 1])[:, None, None, None] - ind).astype(np.float32)
    nrm = vec / (np.linalg.norm(vec, axis=0, keepdims=True) + 1e-12)
    return np.maximum(d - 0.5 * dx, 0.0).astype(np.float32), np.moveaxis(nrm, 0, -1)


def vorticity_magnitude(U: np.ndarray, dx: float = DX) -> np.ndarray:
    g = [np.gradient(U[i], dx) for i in range(3)]          # g[i][j] = dU_i / dx_j
    return np.sqrt((g[2][1] - g[1][2]) ** 2 + (g[0][2] - g[2][0]) ** 2 + (g[1][0] - g[0][1]) ** 2)


def read_voxel_npz(path: str) -> dict:
    with np.load(path) as z:
        return dict(inp=z["inp"], tgt=z["tgt"], mask=z["mask"][0].astype(bool), order=z["order"],
                    info=json.loads(str(z["info"])))


def _to_channels(a: np.ndarray, fluid: np.ndarray, Lz: float, L_ref: float) -> np.ndarray:
    """(6, grid) cache channels -> (n_fluid, 6) canonical [U, Cp, k, eps]."""
    a = a.astype(np.float32)
    return np.stack([a[0][fluid], a[1][fluid], a[2][fluid], 2.0 * a[3][fluid],
                     np.expm1(np.clip(a[4][fluid], 0, None)),
                     np.expm1(np.clip(a[5][fluid], 0, None)) * (L_ref / Lz)], 1)


def voxel_to_case(v: dict, split: str = "", dx: float = DX, with_prior: bool = True, return_vort: bool = False,
                  nu: float | None = None):
    """`nu` (kinematic viscosity from the case's params.json) sets Re = U_ref H / nu; without it Re is absent."""
    fluid = v["mask"]; solid = ~fluid; info = v["info"]
    Lz = float(info["Lz"])
    H = mean_building_height(solid, dx)
    L_ref = H if np.isfinite(H) and H > 0 else Lz / 3.0
    origin = footprint_centroid(solid, dx) if solid[:, :, 0].any() else np.zeros(3, np.float32)
    ix, iy, iz = np.nonzero(fluid)
    xyz = (np.stack([ix, iy, iz], 1).astype(np.float32) + 0.5) * dx
    d, nrm = wall_features(fluid, dx)
    fields = _to_channels(v["tgt"], fluid, Lz, L_ref)
    prior = _to_channels(v["inp"][0:6], fluid, Lz, L_ref) if with_prior else None
    cond = Conditioning(log10_re=float(np.log10(float(info["Uref"]) * L_ref / nu)) if nu else None,
                        closure="k-epsilon", abl_alpha=float(info["alpha"]),
                        turb_intensity=float(v["inp"][10][fluid].astype(np.float32).mean()),
                        ground="stationary", inflow_dir=(1.0, 0.0, 0.0), has_ground=True, rotation_ok=True,
                        prior_kind="coarse-twin" if with_prior else "none")
    meta = dict(tier=info.get("tier"), grid=info.get("grid"), n_cells=int(fluid.sum()), morphology=info.get("morphology"),
                Lz=Lz, H=H, origin_m=origin.tolist(), T_fine=info.get("T_fine"), T_coarse=info.get("T_coarse"),
                converged=bool(info.get("converged_fine", True)), diverged=bool(info.get("diverged", False)),
                solver="OpenFOAM foamRun incompressibleFluid, steady SIMPLE", fidelity="RANS",
                mesh="Cartesian voxel dx=0.05 m (blockMesh+subsetMesh)", prior="coarse twin 2dx, nearest-upsampled",
                split=split, nu=nu, wall_dist_method="voxel EDT to nearest solid centre minus dx/2",
                pressure_gauge="kinematic pressure, fixedValue 0 on the outlet", p_ref=0.0)
    case = Case(case_id=f"aero_urban/{info.get('tier')}/{info['case']}", source="aero_urban",
                points=(xyz - origin) / L_ref, wall_dist=d[fluid] / L_ref, normal=nrm[fluid].astype(np.float32),
                fields=fields, presence=np.array([True, True, True, True]), cond=cond, L_ref=L_ref,
                U_ref=float(info["Uref"]), cell_size=np.full(xyz.shape[0], dx / L_ref, np.float32),
                prior=prior, prior_presence=np.array([True, True, True, True]) if with_prior else None, meta=meta)
    if return_vort:
        return case, vorticity_magnitude(v["tgt"][0:3].astype(np.float32), dx)[fluid]
    return case
