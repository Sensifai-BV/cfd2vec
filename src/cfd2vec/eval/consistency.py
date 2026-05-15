"""Consistency checks that need real cases (acceptance tests of the V1 plan that the unit suite cannot cover):

    compare_frames        one physical case read through two geometry paths (voxel cache versus solver adapter,
                          or two adapters) must give one canonical frame: same L_ref, same origin, same wall
                          distance at the same physical location
    compare_on_shared_points
                          two predictions of one physical problem on two meshes (a remesh or a refinement) are
                          compared at shared physical locations, and each against the truth where it exists

Both work in physical coordinates recovered from the canonical frame (`Case.meta` origin and `L_ref`), so the
frames under test never enter the matching. Nothing here trains or modifies anything.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..schema import Case
from .metrics import rel_l2


def physical_origin(case: Case) -> np.ndarray:
    """Origin of the canonical frame in the source's own coordinates (adapters record `origin`, the voxel source
    `origin_m`)."""
    o = case.meta.get("origin", case.meta.get("origin_m"))
    if o is None:
        raise ValueError(f"{case.case_id}: meta carries no origin; cannot recover physical coordinates")
    return np.asarray(o, np.float64)


def physical_points(case: Case) -> np.ndarray:
    return case.points.astype(np.float64) * float(case.L_ref) + physical_origin(case)


def physical_fields(f: np.ndarray, case: Case):
    """Normalised [U, Cp, k, eps] -> physical [U, kinematic p, k, eps] with the case's own U_ref, L_ref and pressure
    gauge (`meta["p_ref"]`; taken as 0 and flagged when the case records none). Returns (fields, p_ref_assumed)."""
    U, L = float(case.U_ref), float(case.L_ref)
    p_ref = case.meta.get("p_ref")
    out = np.array(f, np.float64, copy=True)
    out[:, 0:3] *= U
    out[:, 3] = 0.5 * out[:, 3] * U ** 2 + (0.0 if p_ref is None else float(p_ref))
    out[:, 4] *= U ** 2
    out[:, 5] *= U ** 3 / L
    return out, p_ref is None


def _q(x, qs=(0.5, 0.9, 0.99)) -> dict:
    x = np.asarray(x)
    if x.size == 0:
        return {f"q{int(100 * q)}": None for q in qs}
    return {f"q{int(100 * q)}": float(v) for q, v in zip(qs, np.quantile(x, qs))}


def compare_frames(a: Case, b: Case, n: int = 20000, seed: int = 0) -> dict:
    """Frame agreement of two readings of one physical case.

    Reports L_ref and origin of both readings with their differences, and, at `n` random points of `a` matched to
    the nearest point of `b` in physical coordinates, the matching distance and the absolute wall-distance
    difference (both in units of `a.L_ref`). A correct pair gives L_ref and origin differences at the level of the
    footprint discretisation and wall-distance differences at the level of the wall-face sampling of each path."""
    pa, pb = physical_points(a), physical_points(b)
    rng = np.random.default_rng(seed)
    ia = rng.choice(a.n, min(int(n), a.n), replace=False)
    d, j = cKDTree(pb).query(pa[ia], workers=-1)
    wd_a, wd_b = a.wall_dist[ia] * float(a.L_ref), b.wall_dist[j] * float(b.L_ref)      # metres
    oa, ob = physical_origin(a), physical_origin(b)
    return dict(n=int(len(ia)), L_ref=[float(a.L_ref), float(b.L_ref)],
                L_ref_rel_diff=float(abs(a.L_ref - b.L_ref) / a.L_ref),
                origin=[oa.tolist(), ob.tolist()], origin_diff_over_L=float(np.linalg.norm(oa - ob) / a.L_ref),
                match_distance_over_L=_q(d / a.L_ref), wall_dist_abs_diff_over_L=_q(np.abs(wd_a - wd_b) / a.L_ref))


def compare_on_shared_points(pred_a: np.ndarray, a: Case, pred_b: np.ndarray, b: Case, n: int = 20000,
                             seed: int = 0, max_match_over_L: float | None = None) -> dict:
    """Agreement of two predictions (normalised fields, rows aligned with `a.points` and `b.points`) of one physical
    problem on two meshes. Points of `a` are matched to the nearest points of `b` in physical coordinates and the
    fields are converted to physical units with each case's own U_ref, L_ref and pressure gauge before comparison
    (pressure relative to `a`'s gauge), so cases normalised differently compare correctly; two cases that record
    different pressure-gauge conventions are rejected. The relative L2 between the two predictions per channel group
    is reported at the matched points, and each prediction against the truth of `a` when `a` carries fields. With
    `max_match_over_L`, pairs farther apart than that (in units of a.L_ref) are dropped, so a coarse mesh is only
    compared where it has a point nearby; when that leaves no pair the result has `status: insufficient_matches`,
    the sampled and retained counts and undefined (None) metrics, so an acceptance report can record the failure."""
    if pred_a.shape[0] != a.n or pred_b.shape[0] != b.n:
        raise ValueError("predictions must have one row per point of their case")
    ga, gb = a.meta.get("pressure_gauge"), b.meta.get("pressure_gauge")
    if ga is not None and gb is not None and ga != gb:
        raise ValueError(f"pressure gauges differ: {ga!r} vs {gb!r}; convert both cases to one convention first")
    fa, pa_assumed = physical_fields(pred_a, a)
    fb, pb_assumed = physical_fields(pred_b, b)
    p0 = 0.0 if a.meta.get("p_ref") is None else float(a.meta["p_ref"])
    fa[:, 3] -= p0; fb[:, 3] -= p0                                  # one gauge for the relative pressure error
    pa, pb = physical_points(a), physical_points(b)
    rng = np.random.default_rng(seed)
    ia = rng.choice(a.n, min(int(n), a.n), replace=False)
    d, j = cKDTree(pb).query(pa[ia], workers=-1)
    n_sampled, d_all = int(len(ia)), d.copy()
    if max_match_over_L is not None:
        keep = d / a.L_ref <= max_match_over_L
        ia, j, d = ia[keep], j[keep], d[keep]
    out = dict(status="ok", n=int(len(ia)), n_sampled=n_sampled, n_retained=int(len(ia)),
               max_match_over_L=max_match_over_L,
               match_distance_over_L=_q(d / a.L_ref), sampled_match_distance_over_L=_q(d_all / a.L_ref),
               units="physical: U, kinematic p relative to case a's p_ref, k, eps",
               scales=dict(a=dict(U_ref=float(a.U_ref), L_ref=float(a.L_ref), p_ref=a.meta.get("p_ref")),
                           b=dict(U_ref=float(b.U_ref), L_ref=float(b.L_ref), p_ref=b.meta.get("p_ref"))),
               p_ref_assumed_zero=[bool(pa_assumed), bool(pb_assumed)],
               between_predictions=None, a_vs_truth=None, b_vs_truth_of_a=None)
    if len(ia) == 0:
        # a strict tolerance can exclude every pair even for overlapping meshes: report that, never a zero error
        out["status"] = "insufficient_matches"
        out["reason"] = (f"none of the {n_sampled} sampled points of a has a point of b within "
                         f"{max_match_over_L} L_ref (nearest: {out['sampled_match_distance_over_L']['q50']:.3g} median)")
        return out
    pres = np.ones(4, bool)
    out["between_predictions"] = rel_l2(fb[j], fa[ia], pres)
    if a.fields is not None:
        ta, _ = physical_fields(a.fields, a); ta[:, 3] -= p0
        out["a_vs_truth"] = rel_l2(fa[ia], ta[ia], a.presence)
        out["b_vs_truth_of_a"] = rel_l2(fb[j], ta[ia], a.presence)
    return out
