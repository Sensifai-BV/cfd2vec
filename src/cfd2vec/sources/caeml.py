"""Volume-file sources: AhmedML, DrivAerML, WindsorML, HiLiftAeroML (caemldatasets layout) and DrivAerNet++.

Ingestion fails closed. `read_run` needs a validated source manifest (configs/source_manifests/<source>.yaml) that
states every convention the conversion depends on: solid-cell mask, Reynolds-stress component order, pressure
convention, inflow direction, reference length, origin, ground plane, licence, and the run that was checked by hand
through `to_canonical` and `from_canonical`. A missing entry, or one still marked VERIFY, raises
`SourceNotValidated` before any `Case` is produced. Recommended: verify one real run per source against its
published integral values, then convert in bulk close to where the data is stored and keep only the fp16 shards.

Assumed layout per run i:  run_i/{volume_i.vtu, boundary_i.vtp, <body>_i.stl, force_mom_i.csv}
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..schema import Case, Conditioning
from ..data.sampling import stratified_indices


@dataclass
class SourceSpec:
    name: str
    closure: str
    fidelity: str
    ground: str
    field_U: str = "UMean"
    field_p: str = "pMean"
    field_k: Optional[str] = None           # modelled k (RANS sources)
    field_R: Optional[str] = "UPrime2Mean"  # resolved Reynolds stress (scale-resolving sources)
    field_ksgs: Optional[str] = None
    rotation_ok: bool = False               # wind-tunnel set-ups have no rotational symmetry
    notes: list = field(default_factory=list)


SPECS = {
    "ahmedml": SourceSpec("ahmedml", "hybrid-RANS-LES", "HRLES", "stationary"),
    "drivaerml": SourceSpec("drivaerml", "hybrid-RANS-LES", "HRLES", "moving",
                            notes=["volume files are ~50 GB: stream-read and subsample in place"]),
    "windsorml": SourceSpec("windsorml", "WMLES", "WMLES", "stationary",
                            notes=["immersed-boundary volume: the solid-cell mask field must be stated"]),
    "hiliftaeroml": SourceSpec("hiliftaeroml", "WMLES", "WMLES", "none",
                               notes=["L_ref is the mean aerodynamic chord; the x-extent rule is not accepted"]),
    "drivaernetpp": SourceSpec("drivaernetpp", "k-omega-SST", "RANS", "moving", field_k="k", field_R=None),
}


def surface_samples(stl_path: str):
    """Triangle centroids and outward unit normals of the body surface."""
    import pyvista as pv
    s = pv.read(stl_path).triangulate().compute_normals(cell_normals=True, point_normals=False,
                                                         auto_orient_normals=True)
    return np.asarray(s.cell_centers().points, np.float64), np.asarray(s.cell_data["Normals"], np.float64), s


def wall_distance_to_surface(points: np.ndarray, surf_pts: np.ndarray, ground_z: Optional[float]):
    """Distance and direction to the nearest surface sample; the ground plane counts as a wall when present.
    Recommended refinement for thin features: exact point-to-triangle distance (pyvista find_closest_cell)."""
    d, j = cKDTree(surf_pts).query(points, workers=-1)
    nrm = points - surf_pts[j]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    if ground_z is not None:
        dg = points[:, 2] - ground_z
        g = dg < d
        d = np.where(g, dg, d); nrm[g] = (0.0, 0.0, 1.0)
    return d, nrm


MANIFEST_KEYS = ("source", "solid_mask", "stress_order", "pressure", "inflow_dir", "L_ref", "origin", "ground_z",
                 "licence", "verified_run")
VERIFY = "VERIFY"
L_REF_VALUE_REQUIRED = {"hiliftaeroml"}          # sources whose reference length is not a bounding-box extent


class SourceNotValidated(ValueError):
    """A source convention required for conversion is missing or unresolved."""


def _unresolved(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "" or VERIFY in v
    if isinstance(v, dict):
        return any(_unresolved(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return len(v) == 0 or any(_unresolved(x) for x in v)
    return False


def validate_manifest(m: dict, source: Optional[str] = None) -> dict:
    """Check a source manifest; returns it unchanged or raises SourceNotValidated naming every problem."""
    problems = [f"{k}: missing" for k in MANIFEST_KEYS if k not in m]
    problems += [f"{k}: unresolved ({m[k]!r})" for k in MANIFEST_KEYS if k in m and _unresolved(m[k])]
    if not problems:
        if source is not None and m["source"] != source:
            problems.append(f"source: manifest is for {m['source']!r}, not {source!r}")
        sm = m["solid_mask"]
        if not (sm == "none" or (isinstance(sm, dict) and "field" in sm and "solid_value" in sm)):
            problems.append("solid_mask: 'none' or {field, solid_value}")
        if m["stress_order"] not in ("vtk", "foam", "none"):
            problems.append("stress_order: one of vtk (XX YY ZZ XY YZ XZ), foam (XX XY XZ YY YZ ZZ), none")
        pr = m["pressure"]
        if not isinstance(pr, dict) or pr.get("convention") not in ("kinematic", "static"):
            problems.append("pressure: {convention: kinematic} or {convention: static, rho, p_ref}")
        elif pr["convention"] == "static" and not {"rho", "p_ref"} <= set(pr):
            problems.append("pressure: static pressure needs rho and p_ref")
        d = np.asarray(m["inflow_dir"], float)
        if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-9:
            problems.append("inflow_dir: a non-zero 3-vector")
        L = m["L_ref"]
        if not (isinstance(L, dict) and (("value" in L and float(L["value"]) > 0) or L.get("rule") == "x-extent")):
            problems.append("L_ref: {value: <length>} or {rule: x-extent}")
        elif m["source"] in L_REF_VALUE_REQUIRED and "value" not in L:
            problems.append("L_ref: this source needs an explicit value (mean aerodynamic chord)")
        o = m["origin"]
        if not (isinstance(o, dict) and (o.get("rule") == "bbox-centre-ground" or
                                         ("point" in o and len(o["point"]) == 3))):
            problems.append("origin: {rule: bbox-centre-ground} or {point: [x, y, z]}")
        if not (m["ground_z"] == "none" or isinstance(m["ground_z"], (int, float))):
            problems.append("ground_z: a height or 'none'")
    if problems:
        raise SourceNotValidated(f"{m.get('source', source)}: " + "; ".join(problems))
    return m


def load_manifest(path: str, source: Optional[str] = None) -> dict:
    import yaml
    return validate_manifest(yaml.safe_load(open(path)), source)


def k_from_stress(R: np.ndarray, order: str) -> np.ndarray:
    if order == "vtk":                      # XX YY ZZ XY YZ XZ
        tr = R[:, 0] + R[:, 1] + R[:, 2]
    elif order == "foam":                   # XX XY XZ YY YZ ZZ
        tr = R[:, 0] + R[:, 3] + R[:, 5]
    else:
        raise ValueError(f"stress order {order!r}")
    return 0.5 * tr


def to_canonical(U: np.ndarray, p: np.ndarray, k: np.ndarray, U_ref: float, pressure: dict) -> np.ndarray:
    """Source fields -> canonical (N, 6) [U / U_ref, Cp, k / U_ref^2, eps = 0]."""
    if pressure["convention"] == "kinematic":
        cp = 2.0 * p / U_ref ** 2
    else:
        cp = 2.0 * (p - float(pressure["p_ref"])) / (float(pressure["rho"]) * U_ref ** 2)
    f = np.zeros((len(U), 6), np.float32)
    f[:, 0:3] = U / U_ref; f[:, 3] = cp; f[:, 4] = k / U_ref ** 2
    return f


def from_canonical(f: np.ndarray, U_ref: float, pressure: dict) -> tuple:
    """Inverse of `to_canonical`: (U, p, k) in the source's own units and pressure convention."""
    U = f[:, 0:3].astype(np.float64) * U_ref
    if pressure["convention"] == "kinematic":
        p = f[:, 3].astype(np.float64) * U_ref ** 2 / 2.0
    else:
        p = f[:, 3].astype(np.float64) * float(pressure["rho"]) * U_ref ** 2 / 2.0 + float(pressure["p_ref"])
    return U, p, f[:, 4].astype(np.float64) * U_ref ** 2


def reference_frame(surf_bounds, manifest: dict, ground_z: Optional[float]):
    lo, hi = np.asarray(surf_bounds[0::2], float), np.asarray(surf_bounds[1::2], float)
    L = manifest["L_ref"]
    L_ref = float(L["value"]) if "value" in L else float(hi[0] - lo[0])
    o = manifest["origin"]
    if "point" in o:
        origin = np.asarray(o["point"], float)
    else:
        origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, ground_z if ground_z is not None else lo[2]])
    return L_ref, origin


def read_run(run_dir: str, run_id: int, spec: SourceSpec, U_ref: float, nu: Optional[float], manifest: dict,
             n_keep: int = 1_500_000, seed: int = 0, split: str = "", family: Optional[str] = None) -> Case:
    """Volume VTU (cell data) -> canonical Case, stratified to n_keep points. `manifest` must pass
    `validate_manifest`; every convention used below comes from it."""
    import pyvista as pv
    m = validate_manifest(manifest, spec.name)
    vtu = glob.glob(os.path.join(run_dir, f"volume_{run_id}.vtu"))[0]
    stl = sorted(glob.glob(os.path.join(run_dir, "*.stl")))[0]
    mesh = pv.read(vtu); cd = mesh.cell_data
    pts = np.asarray(mesh.cell_centers().points, np.float64)
    keep = np.ones(len(pts), bool)
    if m["solid_mask"] != "none":
        name = m["solid_mask"]["field"]
        if name not in cd:
            raise SourceNotValidated(f"{spec.name}: solid-mask field {name!r} not found in {vtu}")
        keep &= np.asarray(cd[name]) != m["solid_mask"]["solid_value"]
    for name in (spec.field_U, spec.field_p):
        if name not in cd:
            raise SourceNotValidated(f"{spec.name}: required field {name!r} not found in {vtu}")
    pts = pts[keep]
    U = np.asarray(cd[spec.field_U], np.float64)[keep]
    p = np.asarray(cd[spec.field_p], np.float64)[keep]
    has_k = True
    if spec.field_k and spec.field_k in cd:
        k = np.asarray(cd[spec.field_k], np.float64)[keep]
    elif spec.field_R and spec.field_R in cd and m["stress_order"] != "none":
        k = k_from_stress(np.asarray(cd[spec.field_R], np.float64)[keep], m["stress_order"])
        if spec.field_ksgs and spec.field_ksgs in cd:
            k = k + np.asarray(cd[spec.field_ksgs], np.float64)[keep]
    else:
        k = np.zeros(len(pts)); has_k = False
    ground_z = None if m["ground_z"] == "none" else float(m["ground_z"])
    sp, _, surf = surface_samples(stl)
    L_ref, origin = reference_frame(surf.bounds, m, ground_z)
    d, nrm = wall_distance_to_surface(pts, sp, ground_z)
    rng = np.random.default_rng(seed)
    inflow = np.asarray(m["inflow_dir"], float); inflow = inflow / np.linalg.norm(inflow)
    deficit = np.linalg.norm(U - U_ref * inflow, axis=1)                  # wake indicator
    idx, strat = stratified_indices(d / L_ref, deficit, n_keep, rng)
    f = to_canonical(U[idx], p[idx], k[idx], U_ref, m["pressure"])
    cond = Conditioning(log10_re=float(np.log10(U_ref * L_ref / nu)) if nu else None, closure=spec.closure,
                        ground=spec.ground, inflow_dir=tuple(inflow.tolist()), has_ground=ground_z is not None,
                        rotation_ok=spec.rotation_ok)
    return Case(case_id=f"{spec.name}/run_{run_id}", source=spec.name,
                points=((pts[idx] - origin) / L_ref).astype(np.float32), wall_dist=(d[idx] / L_ref).astype(np.float32),
                normal=nrm[idx].astype(np.float32), fields=f, presence=np.array([True, True, has_k, False]),
                cond=cond, L_ref=L_ref, U_ref=U_ref, stratum=strat,
                meta=dict(fidelity=spec.fidelity, n_cells=int(keep.sum()), split=split, family=family,
                          validated=True, licence=m["licence"], verified_run=m["verified_run"],
                          wall_dist_method="nearest triangle centroid", L_ref_rule=m["L_ref"],
                          origin_rule=m["origin"], pressure=m["pressure"], stress_order=m["stress_order"]))
