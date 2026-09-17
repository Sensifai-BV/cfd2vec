import numpy as np

from cfd2vec.data.sampling import context_indices, mirror_case_y, rotate_case_z, stratified_indices, subsample
from cfd2vec.schema import Case, decode_fields, encode_fields


def test_shard_roundtrip(tmp_path, case):
    p = str(tmp_path / "c.npz"); case.save(p)
    c = Case.load(p)
    assert c.n == case.n and c.cond == case.cond
    np.testing.assert_allclose(c.fields, case.fields, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(c.prior, case.prior, atol=2e-3, rtol=2e-3)


def test_field_transform_inverse(case):
    np.testing.assert_allclose(decode_fields(encode_fields(case.fields)), case.fields, rtol=1e-5, atol=1e-7)


def test_rotation_is_consistent(case):
    th = 0.7
    r = rotate_case_z(case, th)
    # invariants
    np.testing.assert_allclose(r.wall_dist, case.wall_dist)
    np.testing.assert_allclose(r.fields[:, 3:], case.fields[:, 3:])
    np.testing.assert_allclose(np.linalg.norm(r.fields[:, :3], axis=1), np.linalg.norm(case.fields[:, :3], axis=1), rtol=1e-5)
    # streamwise coordinate and velocity component along the inflow are invariant
    np.testing.assert_allclose(r.streamwise(), case.streamwise(), atol=1e-5)
    d = np.asarray(r.cond.inflow_dir)
    np.testing.assert_allclose(r.fields[:, :3] @ d, case.fields[:, 0], atol=1e-5)
    # prior rotates with the field
    np.testing.assert_allclose(r.prior[:, :3] @ d, case.prior[:, 0], atol=1e-5)
    # full turn is identity
    np.testing.assert_allclose(rotate_case_z(r, -th).points, case.points, atol=1e-5)


def test_mirror(case):
    m = mirror_case_y(case)
    np.testing.assert_allclose(m.points[:, 1], -case.points[:, 1])
    np.testing.assert_allclose(m.fields[:, 1], -case.fields[:, 1])
    np.testing.assert_allclose(mirror_case_y(m).fields, case.fields)


def test_stratified_quota(case):
    rng = np.random.default_rng(0)
    vort = np.abs(case.fields[:, 1]) + rng.random(case.n) * 1e-3
    idx, s = stratified_indices(case.wall_dist, vort, 1000, rng, near_wall_band=1.0)
    assert np.unique(idx).size == idx.size
    assert (s == 1).sum() == 500 and (s == 0).sum() == 200 and 0 < (s == 2).sum() <= 300
    sub = subsample(case, idx)
    assert sub.n == idx.size and sub.prior.shape == (idx.size, 6)


def test_context_is_geometry_only(case):
    """Context sampling must not depend on the solution: two different flows on one geometry give one context."""
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    a = context_indices(case.wall_dist, 800, rng1)
    b = context_indices(case.wall_dist, 800, rng2)
    np.testing.assert_array_equal(a, b)
    near = (case.wall_dist[a] < 0.25).mean()
    assert near <= 5 / 7 + 1e-6
