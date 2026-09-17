"""Wall-normal orientation, frame and remesh consistency, channel-statistics options, tokeniser statistics, the
fine-tuning protocol's augmentation switches and schedule, checkpoint lineage, and data-path expansion."""
import copy
import math
import os

import numpy as np
import pytest
import torch

from cfd2vec.data.sampling import subsample
from cfd2vec.train.dataset import channel_stats, make_sample, tokeniser_stats

from conftest import synthetic_case
from test_model import TINY, spec_for


# ----------------------------------------------------------------------------------------------- wall normals
def test_orientation_uses_owner_connectivity_only_and_never_a_distance_heuristic():
    """The technical review's stretched-cell example: the adjacent fluid cell above a slab face is 0.1 x 0.1 x 2 with
    its centre 1.0 away, a cell beneath the slab is 0.2 away. Without connectivity the normal must be returned as
    supplied and marked unverified; with the owner it is oriented exactly."""
    from cfd2vec.solvers.base import orient_normals_to_fluid
    fc, fn = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0]])
    out, rep = orient_normals_to_fluid(fc, fn)
    np.testing.assert_array_equal(out, fn)
    assert rep["method"] == "reader" and rep["verified"] is False and rep["flipped"] == 0.0
    out, rep = orient_normals_to_fluid(fc, fn, owner_centres=np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_array_equal(out, fn); assert rep["method"] == "owner" and rep["verified"] and rep["flipped"] == 0.0
    out, rep = orient_normals_to_fluid(fc, -fn, owner_centres=np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_array_equal(out, fn); assert rep["flipped"] == 1.0
    with pytest.raises(ValueError):
        orient_normals_to_fluid(fc, fn, owner_centres=np.zeros((2, 3)))


def test_polymesh_header_keeps_quoted_values_and_honours_widths(tmp_path):
    """The reviewer's reproduction: `arch "LSB;label=64;scalar=32"` must not be cut at its first semicolon. Labels and
    points written with those widths (and in big-endian order) read back with their actual values; an unknown
    declaration raises instead of guessing."""
    from cfd2vec.solvers import polymesh as pm
    from foam_fixture import _header
    p = str(tmp_path / "owner")
    with open(p, "wb") as f:
        f.write(_header("binary", "labelList", "owner", "LSB;label=64;scalar=32").encode())
        f.write(b"2\n("); f.write(np.array([0, 1], "<i8").tobytes()); f.write(b")\n")
    np.testing.assert_array_equal(pm.read_labels(p), [0, 1])
    q = str(tmp_path / "points")
    with open(q, "wb") as f:
        f.write(_header("binary", "vectorField", "points", "MSB;label=64;scalar=32").encode())
        f.write(b"2\n("); f.write(np.array([[1, 2, 3], [4.5, 5, 6]], ">f4").tobytes()); f.write(b")\n")
    np.testing.assert_allclose(pm.read_points(q), [[1, 2, 3], [4.5, 5, 6]])
    h, _ = pm._header(open(q, "rb").read())
    assert h["arch"] == "MSB;label=64;scalar=32" and pm._dtypes(h) == (">i8", ">f4", 8, 4)
    with pytest.raises(ValueError, match="arch"):
        pm._dtypes({"arch": "LSB;label=48;scalar=64"})


@pytest.mark.parametrize("fmt", ["ascii", "LSB;label=32;scalar=64", "LSB;label=64;scalar=32", "MSB;label=64;scalar=64"])
def test_polymesh_reader_and_owner_verification(tmp_path, fmt):
    """The polyMesh files of the slab fixture (ascii, and binary in three label / scalar / byte-order variants) parse
    to the generator's geometry, and reader faces given by their centres are matched one to one to polyMesh faces
    with the owner cell established from topology: roof faces are owned by the stretched cells (centre z = 2),
    floor faces by the fine cells below (z = 0.85). Values are checked, not only headers."""
    from cfd2vec.solvers import polymesh as pm
    from foam_fixture import slab_case
    binary = fmt != "ascii"
    kw = {}
    if binary:
        order, lab, sc = fmt.split(";"); kw = dict(order=order, label=int(lab.split("=")[1]), scalar=int(sc.split("=")[1]))
    d = str(tmp_path / "case"); ref = slab_case(d, binary=binary, **kw)
    mesh = pm.read_polymesh(os.path.join(d, "constant", "polyMesh"))
    assert mesh["face_offsets"].size - 1 == ref["n_faces"] and mesh["neighbour"].size == ref["n_internal"]
    assert set(mesh["patches"]) == set(ref["patches"]) and mesh["patches"]["building"]["type"] == "wall"
    cc = pm.cell_centres(mesh)
    np.testing.assert_allclose(cc, ref["cell_centres"], atol=1e-9)         # hex cells: face-centre mean is exact
    rf = [ref["patches"][w]["centres"] for w in ("ground", "building")]
    widths = [np.full(len(x), 0.1) for x in rf]
    oc, rep = pm.verified_owner_centres(d, ["ground", "building"], rf, widths)
    want = np.concatenate([ref["patches"][w]["owner_centres"] for w in ("ground", "building")])
    np.testing.assert_allclose(oc, want, atol=1e-9)
    assert rep["verified"] and rep["matched_faces"] == len(want) and rep["polymesh_format"] == ("binary" if binary else "ascii")
    if binary:                                                              # the declared widths were honoured
        h, _ = pm._header(open(os.path.join(d, "constant", "polyMesh", "owner"), "rb").read())
        assert h["arch"] == fmt
    roof = ref["patches"]["building"]["into_fluid"][:, 2] > 0.5
    assert np.allclose(oc[len(rf[0]):][roof, 2], 2.0) and np.allclose(oc[len(rf[0]):][~roof & (ref["patches"]["building"]["into_fluid"][:, 2] < -0.5), 2], 0.85)
    with pytest.raises(ValueError, match="reader faces"):                 # a count mismatch does not verify
        pm.verified_owner_centres(d, ["building"], [rf[1][:-1]], [widths[1][:-1]])
    moved = rf[1].copy(); moved[0] += 0.05                                  # a face that is not at a polyMesh face
    with pytest.raises(ValueError, match="face order or geometry"):
        pm.verified_owner_centres(d, ["building"], [moved], [widths[1]])
    with pytest.raises(ValueError, match="not a patch"):
        pm.verified_owner_centres(d, ["nothere"], [rf[1]], [widths[1]])


def test_remesh_comparison_reports_insufficient_matches_instead_of_crashing():
    """The reviewer's reproduction: a strict match tolerance can exclude every sampled pair; the result must say so
    with the counts and undefined metrics, never a zero error."""
    from cfd2vec.eval.consistency import compare_on_shared_points
    a = synthetic_case(n=1000); a.meta["origin"] = [0.0, 0.0, 0.0]
    b = copy.copy(a); b.points = (a.points + 0.01).astype(np.float32); b.meta = dict(a.meta)
    r = compare_on_shared_points(a.fields, a, b.fields, b, n=500, max_match_over_L=1e-6)
    assert r["status"] == "insufficient_matches" and r["n_retained"] == 0 and r["n_sampled"] == 500
    assert r["between_predictions"] is None and r["a_vs_truth"] is None and "reason" in r
    assert r["match_distance_over_L"]["q50"] is None and r["sampled_match_distance_over_L"]["q50"] > 0.01
    ok = compare_on_shared_points(a.fields, a, b.fields, b, n=500, max_match_over_L=0.1)
    assert ok["status"] == "ok" and ok["n_retained"] == 500 and ok["between_predictions"]["rel_l2_U"] == 0.0


@pytest.mark.parametrize("reverse", [False, True])
def test_openfoam_adapter_orients_from_owner_connectivity_through_the_reader(tmp_path, reverse):
    """The slab fixture through the actual OpenFOAM reader: orientation is established from the polyMesh, the roof
    normals point up whether or not the mesh file lists the wall faces reversed, and the footprint normalisation
    (L_ref = slab height, origin = slab footprint centroid) follows."""
    pytest.importorskip("pyvista"); pytest.importorskip("foamlib")
    from cfd2vec.schema import Conditioning
    from cfd2vec.solvers.openfoam import OpenFOAMAdapter
    from foam_fixture import slab_case
    d = str(tmp_path / "case"); slab_case(d, reverse_wall_faces=reverse)
    case = OpenFOAMAdapter().read_case(d, U_ref=1.0, L_ref=None, cond=Conditioning(closure="k-epsilon"))
    o = case.meta["normal_orientation"]
    assert o["method"] == "owner" and o["verified"] is True, o
    n_building, n_ground = 9 + 9 + 12, 25                                   # slab roof, floor, sides; ground 5 x 5
    assert o["n_faces"] == n_building + n_ground
    assert abs(o["flipped"] - (n_building / (n_building + n_ground) if reverse else 0.0)) < 1e-9, o
    assert abs(case.meta["mean_height"] - 1.0) < 1e-9                        # roof faces are the upward faces
    np.testing.assert_allclose(case.meta["origin"], [0.25, 0.25, 0.0], atol=1e-9)
    assert case.n == 5 * 5 * 10 + 25 - 9 and abs(case.L_ref - 1.0) < 1e-9


def test_vtk_adapter_keeps_surface_orientation_and_reports_unverified(tmp_path):
    pytest.importorskip("pyvista")
    import pyvista as pv
    from cfd2vec.schema import Conditioning
    from cfd2vec.solvers.vtk import VTKAdapter
    grid = pv.ImageData(dimensions=(7, 7, 7), spacing=(0.5, 0.5, 0.5), origin=(-1.5, -1.5, 0.0)).cast_to_unstructured_grid()
    grid.save(str(tmp_path / "mesh.vtu"))
    pv.Cube(center=(0.0, 0.0, 0.5), x_length=1.0, y_length=1.0, z_length=1.0).triangulate().save(str(tmp_path / "walls.stl"))
    case = VTKAdapter().read_case(str(tmp_path), U_ref=1.0, L_ref=None, cond=Conditioning(), ground_z=0.0)
    o = case.meta["normal_orientation"]
    assert o["method"] == "reader" and o["verified"] is False and o["flipped"] == 0.0
    assert abs(case.meta.get("origin", [0, 0, 0])[2]) < 1e-9 and case.L_ref > 0


# ----------------------------------------------------------------------------------------------- consistency
def test_compare_frames_matches_in_physical_coordinates():
    from cfd2vec.eval.consistency import compare_frames
    a = synthetic_case(n=3000); a.meta["origin"] = [10.0, -5.0, 0.0]
    phys = a.points.astype(np.float64) * a.L_ref + np.array([10.0, -5.0, 0.0])
    b = copy.copy(a); ob = np.array([11.0, -4.0, 0.0])
    b.L_ref = 2.0; b.points = ((phys - ob) / 2.0).astype(np.float32); b.wall_dist = (a.wall_dist / 2.0).astype(np.float32)
    b.meta = dict(a.meta, origin=ob.tolist())
    r = compare_frames(a, b, n=1000)
    assert r["n"] == 1000 and r["L_ref_rel_diff"] == 1.0 and abs(r["origin_diff_over_L"] - math.sqrt(2)) < 1e-9
    assert r["match_distance_over_L"]["q99"] < 1e-4 and r["wall_dist_abs_diff_over_L"]["q99"] < 1e-4
    c = copy.copy(a); c.meta = {}
    with pytest.raises(ValueError, match="origin"):
        compare_frames(a, c)


def test_compare_on_shared_points_converts_reference_scales_and_checks_gauges():
    """The technical review's reproduction: identical physical fields normalised with different U_ref (and L_ref) must
    agree; without conversion the velocity would read as a 50 % discrepancy. Different gauge conventions are refused."""
    from cfd2vec.eval.consistency import compare_on_shared_points
    from cfd2vec.eval.metrics import rel_l2
    a = synthetic_case(n=2000)
    a.meta.update(origin=[0.0, 0.0, 0.0], p_ref=0.0, pressure_gauge="kinematic pressure relative to p_ref")
    b = copy.copy(a); b.U_ref = 2.0; b.meta = dict(a.meta)
    fb = a.fields.copy(); fb[:, 0:3] /= 2; fb[:, 3] /= 4; fb[:, 4] /= 4; fb[:, 5] /= 8   # same physical fields
    assert abs(rel_l2(fb, a.fields)["rel_l2_U"] - 0.5) < 1e-6                             # naive comparison
    r = compare_on_shared_points(a.fields, a, fb, b, n=2000)
    for g in ("U", "Cp", "k", "eps"):
        assert r["between_predictions"][f"rel_l2_{g}"] < 1e-6, g
    assert r["p_ref_assumed_zero"] == [False, False] and r["scales"]["b"]["U_ref"] == 2.0
    c = copy.copy(a); c.L_ref = 2.0; c.points = (a.points / 2).astype(np.float32); c.meta = dict(a.meta)
    fc = a.fields.copy(); fc[:, 5] *= 2                                                   # eps* = eps L_ref / U_ref^3
    r2 = compare_on_shared_points(a.fields, a, fc, c, n=2000)
    assert r2["between_predictions"]["rel_l2_eps"] < 1e-6 and r2["match_distance_over_L"]["q99"] < 1e-5
    d = copy.copy(a); d.meta = dict(a.meta, pressure_gauge="static pressure")
    with pytest.raises(ValueError, match="gauge"):
        compare_on_shared_points(a.fields, a, a.fields, d)
    e = copy.copy(a); e.meta = dict(origin=[0.0, 0.0, 0.0])                               # no gauge recorded
    assert compare_on_shared_points(a.fields, a, a.fields, e, n=500)["p_ref_assumed_zero"] == [False, True]


def test_compare_on_shared_points_identity_on_a_submesh():
    from cfd2vec.eval.consistency import compare_on_shared_points
    a = synthetic_case(n=3000); a.meta["origin"] = [0.0, 0.0, 0.0]
    b = subsample(a, np.arange(0, 3000, 3)); b.meta = dict(a.meta)
    r = compare_on_shared_points(a.fields, a, b.fields, b, n=3000, max_match_over_L=1e-6)
    assert r["n"] == 1000 and r["between_predictions"]["rel_l2_U"] == 0.0 and r["a_vs_truth"]["rel_l2_Cp"] == 0.0
    r2 = compare_on_shared_points(a.fields, a, b.fields * 1.1, b, n=3000)
    assert r2["n"] == 3000 and r2["between_predictions"]["rel_l2_U"] > 0.0
    with pytest.raises(ValueError):
        compare_on_shared_points(a.fields[:10], a, b.fields, b)


# ----------------------------------------------------------------------------------------------- statistics
@pytest.fixture
def shards(tmp_path):
    paths = []
    for i in range(3):
        p = str(tmp_path / f"c{i}.npz"); synthetic_case(n=2500, seed=i).save(p); paths.append(p)
    return paths


def test_channel_stats_velocity_scale_option(shards):
    per = channel_stats(shards, n_per_case=1024, velocity_scale="per_channel")
    com = channel_stats(shards, n_per_case=1024, velocity_scale="common")
    assert per["velocity_scale"] == "per_channel" and com["velocity_scale"] == "common"
    assert per["std"][0] == per["std"][1] and per["mean"][0] == per["mean"][1] == 0.0
    assert per["std"][2] != per["std"][0]                        # Uz keeps its own scale
    assert com["std"][0] == com["std"][1] == com["std"][2] == per["std"][0] and com["mean"][2] == 0.0
    assert com["std"][3:] == per["std"][3:] and com["mean"][3:] == per["mean"][3:]
    with pytest.raises(ValueError):
        channel_stats(shards, velocity_scale="isotropic")


def test_tokeniser_stats_report_support_per_scale(shards):
    ts = tokeniser_stats(shards, spec_for(TINY), n_cases=2)
    m = ts["median"]
    assert ts["n_cases"] == 2 and ts["nominal_r1"] == 0.05 and ts["k_neighbors"] == TINY.k_neighbors
    assert 0.0 < m["coverage"] <= 1.0 and 0.0 < m["coverage_near_wall"] <= 1.0
    for sc in (0, 1):
        assert 0 < m[f"r_eff_scale{sc}_q10"] <= m[f"r_eff_scale{sc}_q50"] <= m[f"r_eff_scale{sc}_q90"]
    assert len(ts["cases"]) == 2 and ts["cases"][0]["n_context"] == TINY.n_context


# ----------------------------------------------------------------------------------------------- protocol
def test_protocol_augmentation_switches_are_independent():
    c = synthetic_case(n=1500)                                       # rotation_ok is True for the synthetic case
    base = make_sample(c, spec_for(TINY, fixed_ratio=1.0, augment=False), np.random.default_rng(3), query_case=c)
    off = make_sample(c, spec_for(TINY, fixed_ratio=1.0, augment=True, rotate=False, mirror=False),
                      np.random.default_rng(3), query_case=c)
    np.testing.assert_array_equal(off["q_pos"], base["q_pos"])
    rotated, mirrored = False, False
    for seed in range(8):
        r = make_sample(c, spec_for(TINY, fixed_ratio=1.0, augment=True, rotate=True, mirror=False),
                        np.random.default_rng(seed), query_case=c)["q_pos"]
        np.testing.assert_allclose(r[:, 2], c.points[:, 2], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(r[:, :2], axis=1), np.linalg.norm(c.points[:, :2], axis=1), atol=1e-4)
        rotated |= not np.allclose(r[:, 0], c.points[:, 0], atol=1e-3)
        m = make_sample(c, spec_for(TINY, fixed_ratio=1.0, augment=True, rotate=False, mirror=True),
                        np.random.default_rng(seed), query_case=c)["q_pos"]
        np.testing.assert_allclose(m[:, 0], c.points[:, 0], atol=1e-6)
        np.testing.assert_allclose(np.abs(m[:, 1]), np.abs(c.points[:, 1]), atol=1e-6)
        mirrored |= np.allclose(m[:, 1], -c.points[:, 1], atol=1e-6)
    assert rotated and mirrored


def test_evaluation_never_runs_after_a_stop_signal():
    from cfd2vec.train.pretrain import should_evaluate
    assert should_evaluate(500, 500, False, False) and should_evaluate(7, 500, True, False)
    assert not should_evaluate(7, 500, False, False)
    assert not should_evaluate(500, 500, True, True) and not should_evaluate(500, 500, False, True)
    assert not should_evaluate(7, 500, True, True)


def test_protocol_lr_factor_matches_frozen_protocol_and_extends_it():
    from cfd2vec.train.finetune import protocol_lr_factor
    for s in (0, 7, 50, 99):
        assert abs(protocol_lr_factor(s, 100) - 0.5 * (1 + math.cos(math.pi * s / 100))) < 1e-12
    assert abs(protocol_lr_factor(0, 100, warmup=10) - 0.1) < 1e-12 and protocol_lr_factor(9, 100, warmup=10) == 1.0
    assert abs(protocol_lr_factor(100, 100, min_ratio=0.1) - 0.1) < 1e-12
    assert protocol_lr_factor(10, 100, warmup=10) == 1.0


def test_finetune_refuses_an_unimplemented_schedule(tmp_path, shards):
    import yaml
    from cfd2vec.api import CFD2vec, DEFAULT_PROTOCOL
    from cfd2vec.model.network import CFD2vecNet
    P = yaml.safe_load(open(DEFAULT_PROTOCOL)); P["schedule"]["name"] = "linear"
    proto = str(tmp_path / "p.yaml"); yaml.safe_dump(P, open(proto, "w"))
    m = CFD2vec(CFD2vecNet(TINY), "cpu"); m.stats = dict(mean=[0.0] * 6, std=[1.0] * 6)
    with pytest.raises(ValueError, match="schedule"):
        m.finetune(shards, proto, max_epochs=1)


# ----------------------------------------------------------------------------------------------- lineage and paths
def test_save_keeps_checkpoint_lineage(tmp_path):
    from cfd2vec.api import CFD2vec
    from cfd2vec.model.network import CFD2vecNet
    m = CFD2vec(CFD2vecNet(TINY), "cpu"); m.stats = dict(mean=[0.0] * 6, std=[1.0] * 6)
    p0 = str(tmp_path / "a.pt")
    torch.save(dict(model=m.net.state_dict(), cfg=TINY.to_dict(), stats=m.stats, step=42, objective="masked",
                    provenance=dict(pool_sha256="abc")), p0)
    m2 = CFD2vec.from_pretrained(p0, device="cpu")
    p1 = str(tmp_path / "b.pt"); m2.save(p1, note="after fine-tuning")
    ck = torch.load(p1, weights_only=False)
    assert ck["step"] == 42 and ck["objective"] == "masked" and ck["provenance"]["pool_sha256"] == "abc"
    assert ck["input_schema"] == 2 and ck["note"] == "after fine-tuning"
    assert CFD2vec.from_pretrained(p1, device="cpu").meta["step"] == 42
    p2 = str(tmp_path / "c.pt"); m.save(p2)                          # a model without lineage saves None fields
    assert torch.load(p2, weights_only=False)["step"] is None


def test_data_path_expansion(monkeypatch):
    from cfd2vec.paths import DEFAULT_DATA_ROOT, data_root, expand, expand_config_paths
    from cfd2vec.train.pretrain import resolve_manifests
    monkeypatch.delenv("CFD2VEC_DATA", raising=False)
    assert data_root() == os.path.expanduser(DEFAULT_DATA_ROOT)
    assert expand("${CFD2VEC_DATA}/x.npz") == os.path.join(data_root(), "x.npz")
    monkeypatch.setenv("CFD2VEC_DATA", "/data/cfd")
    assert expand("${CFD2VEC_DATA}/x") == "/data/cfd/x" and expand("configs/m.yaml") == "configs/m.yaml"
    y = expand_config_paths(dict(shards="${CFD2VEC_DATA}/s", model="configs/m.yaml",
                                 manifests=["${CFD2VEC_DATA}/m.parquet"]))
    assert y["shards"] == "/data/cfd/s" and y["model"] == "configs/m.yaml" and y["manifests"] == ["/data/cfd/m.parquet"]
    assert resolve_manifests(dict(shards="${CFD2VEC_DATA}/s", sources=["a"])) == ["/data/cfd/s/manifest_a.parquet"]
