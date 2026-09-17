"""Mathematical and input contracts of the encoder, decoder and sample builder (exact up to fp32 tolerance)."""
import copy

import numpy as np
import pytest
import torch

from cfd2vec.data.sampling import stratified_indices, strip_targets, subsample
from cfd2vec.model.geometry import build_tokens, farthest_point_sampling
from cfd2vec.model.network import CFD2vecNet, ModelConfig, Rope3D, migrate_state_dict
from cfd2vec.train.dataset import make_sample, numpy_collate, to_torch, validate_case

from conftest import synthetic_case
from test_model import TINY, spec_for

TOK_KEYS = ("tok_centre", "tok_nb", "tok_scale", "tok_radius", "tok_reff", "tok_valid", "nb_valid")


def net(schema=2, seed=0, head_std=0.1):
    torch.manual_seed(seed)
    m = CFD2vecNet(ModelConfig(**{**TINY.to_dict(), "input_schema": schema})).eval()
    if head_std:
        torch.nn.init.normal_(m.head.weight, std=head_std)
    return m


def tb(s):
    return {k: v[None] for k, v in to_torch(s).items()}


def run(m, b, chunk=0):
    with torch.no_grad():
        mem, pos, glob = m.encode(b)
        return m.decode(mem, pos, b, chunk=chunk), glob


def spec(**kw):
    kw.setdefault("fixed_ratio", 1.0); kw.setdefault("augment", False); kw.setdefault("use_prior", False)
    return spec_for(TINY, **kw)


# ----------------------------------------------------------------------------------------------- observation contract
@pytest.mark.parametrize("use_prior", [False, True])
def test_fully_masked_output_ignores_stored_labels(use_prior):
    c = synthetic_case(n=2500)
    other = copy.copy(c)
    other.fields = np.random.default_rng(5).normal(size=c.fields.shape).astype(np.float32) * 10
    other.presence = np.array([True, False, True, False])
    variants = [c, strip_targets(c), other]
    samples = [make_sample(v, spec(use_prior=use_prior), np.random.default_rng(0)) for v in variants]
    for k in samples[0]:                           # geometry-only inputs are identical, targets aside
        if k not in ("presence", "q_field"):
            for s in samples[1:]:
                np.testing.assert_array_equal(np.asarray(samples[0][k]), np.asarray(s[k]), err_msg=k)
    m = net(2)
    outs = [run(m, tb(s)) for s in samples]
    for p, g in outs[1:]:
        torch.testing.assert_close(p, outs[0][0], rtol=0, atol=0)
        torch.testing.assert_close(g, outs[0][1], rtol=0, atol=0)
    # the encoder must not read target presence at all in schema 2
    b = tb(samples[0]); del b["presence"]
    run(m, b)


def test_legacy_schema_reads_target_presence():
    """Schema 1 keeps its historical behaviour: stored labels change the fully masked output."""
    c = synthetic_case(n=2500)
    a = run(net(1), tb(make_sample(c, spec(), np.random.default_rng(0))))[1]
    b = run(net(1), tb(make_sample(strip_targets(c), spec(), np.random.default_rng(0))))[1]
    assert not torch.allclose(a, b)


def test_partial_masking_observed_presence_is_per_point():
    c = synthetic_case(n=2500); c.presence = np.array([True, True, False, True])
    s = make_sample(c, spec(fixed_ratio=0.6), np.random.default_rng(0))
    obs, vis = s["ctx_obs"], s["ctx_vis"]
    np.testing.assert_array_equal(obs, vis[:, None] & c.presence[None, :])


def test_nan_in_absent_group_is_harmless_nan_in_present_group_is_rejected():
    c = synthetic_case(n=1500); c.presence = np.array([True, True, True, False]); c.fields[:, 5] = np.nan
    s = make_sample(c, spec(fixed_ratio=0.6), np.random.default_rng(0))
    assert np.isfinite(s["ctx_field"]).all() and np.isfinite(s["q_field"]).all()
    c.fields[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        make_sample(c, spec(), np.random.default_rng(0))
    c2 = synthetic_case(n=500); c2.U_ref = 0.0
    with pytest.raises(ValueError, match="positive"):
        validate_case(c2)


# ----------------------------------------------------------------------------------------------- padding and permutation
def unpadded(s):
    T, K = int(s["tok_valid"].sum()), int(s["nb_valid"][0].sum())
    n = int(np.max(s["tok_nb"][s["nb_valid"]])) + 1
    u = {k: v for k, v in s.items()}
    for k in ("tok_centre", "tok_scale", "tok_radius", "tok_reff", "tok_valid"):
        u[k] = s[k][:T]
    u["tok_nb"], u["nb_valid"] = s["tok_nb"][:T, :K], s["nb_valid"][:T, :K]
    for k in [k for k in s if k.startswith("ctx_")]:
        u[k] = s[k][:n]
    return u


def test_mixed_size_batch_matches_unpadded_inference():
    """Cases below the token budget (40 points < 96 tokens) and below k (10 points < 16 neighbours) batch with a
    full-size case, and every case's result equals its own unpadded evaluation (nonzero head)."""
    cases = [synthetic_case(n=n, seed=i) for i, n in enumerate((10, 40, 3000))]
    ss = [make_sample(c, spec(use_prior=True), np.random.default_rng(i)) for i, c in enumerate(cases)]
    b = {k: torch.as_tensor(np.asarray(v)) for k, v in to_torch(numpy_collate(ss)).items()}
    m = net(2)
    pb, gb = run(m, b)
    for i, s in enumerate(ss):
        pu, gu = run(m, tb(unpadded(s)))
        torch.testing.assert_close(pb[i:i + 1], pu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gb[i:i + 1], gu, atol=1e-5, rtol=1e-5)
    assert not ss[0]["tok_valid"].all() and not ss[0]["nb_valid"].all()


def test_empty_case_is_rejected():
    c = subsample(synthetic_case(n=100), np.zeros(0, int))
    with pytest.raises(ValueError):
        make_sample(c, spec(), np.random.default_rng(0))


def test_neighbour_and_token_permutation_invariance():
    s = make_sample(synthetic_case(n=3000), spec(use_prior=True), np.random.default_rng(0))
    m = net(2)
    p0, g0 = run(m, tb(s))
    rng = np.random.default_rng(1)
    s1 = dict(s); s1["tok_nb"] = s["tok_nb"][:, rng.permutation(s["tok_nb"].shape[1])]
    p1, g1 = run(m, tb(s1))
    torch.testing.assert_close(p1, p0, atol=1e-5, rtol=1e-5); torch.testing.assert_close(g1, g0, atol=1e-5, rtol=1e-5)
    perm = rng.permutation(len(s["tok_centre"]))
    s2 = dict(s); s2.update({k: s[k][perm] for k in TOK_KEYS})
    p2, g2 = run(m, tb(s2))
    torch.testing.assert_close(p2, p0, atol=1e-5, rtol=1e-5); torch.testing.assert_close(g2, g0, atol=1e-5, rtol=1e-5)


def test_query_order_and_chunking_do_not_matter():
    ctx, qry = synthetic_case(n=3000), synthetic_case(n=300, seed=4)
    s = make_sample(ctx, spec(), np.random.default_rng(0), query_case=qry)
    m = net(2)
    p0, _ = run(m, tb(s), chunk=0)
    p1, _ = run(m, tb(s), chunk=37)
    torch.testing.assert_close(p1, p0, atol=1e-5, rtol=1e-5)
    perm = np.random.default_rng(2).permutation(300)
    s2 = dict(s); s2.update({k: s[k][perm] for k in s if k.startswith("q_")})
    p2, _ = run(m, tb(s2), chunk=64)
    torch.testing.assert_close(p2, p0[:, perm], atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------------------------- priors
@pytest.mark.parametrize("schema", [1, 2])
def test_zero_head_reproduces_available_prior_channels(schema):
    """For every prior group pattern: available channels equal the prior, absent channels equal the encoded
    training mean (0 in standardised space)."""
    m = net(schema, head_std=0)
    group = np.array([0, 0, 0, 1, 2, 3])
    for bits in range(16):
        pat = np.array([(bits >> g) & 1 for g in range(4)], bool)
        c = synthetic_case(n=1500); c.prior_presence = pat
        s = make_sample(c, spec(use_prior=True), np.random.default_rng(0))
        p, _ = run(m, tb(s))
        want = m.standardise(torch.as_tensor(s["q_prior"])[None]) * torch.as_tensor(pat[group], dtype=torch.float32)
        torch.testing.assert_close(p, want, atol=1e-5, rtol=1e-5)


def test_partial_prior_patterns_are_distinguishable_in_schema_2():
    """Two partial priors with equal values but different group flags reach the network differently."""
    c = synthetic_case(n=1500)
    m = net(2)
    outs = []
    for pat in ([True, False, False, False], [True, True, False, False]):
        cc = copy.copy(c); cc.prior = c.prior.copy(); cc.prior[:, 3] = 0.0; cc.prior_presence = np.array(pat)
        outs.append(run(m, tb(make_sample(cc, spec(use_prior=True), np.random.default_rng(0))))[0])
    assert not torch.allclose(outs[0], outs[1])


def test_query_prior_must_match_context_prior():
    ctx, qry = synthetic_case(n=1500), synthetic_case(n=200, seed=3)
    qry.prior_presence = np.array([True, False, True, True])
    with pytest.raises(ValueError, match="prior groups"):
        make_sample(ctx, spec(use_prior=True), np.random.default_rng(0), query_case=qry)
    qry.prior = None
    with pytest.raises(ValueError, match="carry none"):
        make_sample(ctx, spec(use_prior=True), np.random.default_rng(0), query_case=qry)


def test_schema_migration_is_loadable_and_keeps_full_prior_flag():
    m1, m2 = net(1), net(2)
    sd = migrate_state_dict(m1.state_dict(), 1, 2)
    m2.load_state_dict(sd)
    w1, w2 = m1.pointnet[0].weight, m2.pointnet[0].weight
    torch.testing.assert_close(w2[:, :w1.shape[1]], w1)
    assert float(w2[:, w1.shape[1]:].detach().abs().sum()) == 0.0


# ----------------------------------------------------------------------------------------------- sampling contracts
def test_short_pool_strata_do_not_depend_on_the_solution():
    c = synthetic_case(n=2000)
    rng = np.random.default_rng(0)
    a = stratified_indices(c.wall_dist, rng.random(c.n), 5000, np.random.default_rng(1))
    b = stratified_indices(c.wall_dist, rng.random(c.n), 5000, np.random.default_rng(1))
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
    assert (a[1] != 2).all()


def test_complete_legacy_shard_context_ignores_wake_stratum():
    """A shard that kept every cell but carries an old wake stratum: context and tokens must not move when the
    wake stratum (a function of the solution) moves."""
    outs = []
    for seed in (0, 1):
        c = synthetic_case(n=1500)
        c.meta = dict(sampling="stratified", n_cells=c.n)
        c.stratum = np.zeros(c.n, np.uint8)
        c.stratum[np.random.default_rng(seed).choice(c.n, 150, replace=False)] = 2
        outs.append(make_sample(c, spec(fixed_ratio=0.6), np.random.default_rng(0)))
    for k in ("ctx_pos", "ctx_wd") + TOK_KEYS:
        np.testing.assert_array_equal(outs[0][k], outs[1][k], err_msg=k)


def test_augmentation_moves_explicit_queries_with_the_context():
    c = synthetic_case(n=1500)
    s = make_sample(c, spec(augment=True), np.random.default_rng(3), query_case=c)
    np.testing.assert_allclose(s["q_pos"], s["ctx_pos"][:c.n], atol=1e-6)
    assert not np.allclose(s["q_pos"], c.points, atol=1e-3)                 # a nontrivial transform was drawn


def test_fps_unique_centres_with_duplicates_and_input_validation():
    base = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    pts = torch.from_numpy(np.repeat(base, 10, axis=0))
    idx = farthest_point_sampling(pts, 5)
    assert len({tuple(p) for p in pts[idx].numpy().round(6)}) == 5          # every distinct coordinate first
    idx = farthest_point_sampling(pts, 20)
    assert idx.unique().numel() == 20
    c = synthetic_case(n=200)
    for kw in (dict(n1=0), dict(r1=0.0), dict(k=0)):
        args = dict(n1=16, n2=8, k=8, r1=0.05, r2=0.25); args.update(kw)
        with pytest.raises(ValueError):
            build_tokens(c.points, c.wall_dist, **args)
    bad = c.points.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        build_tokens(bad, c.wall_dist, 16, 8, 8, 0.05, 0.25)
    with pytest.raises(ValueError):
        build_tokens(np.zeros((0, 3), np.float32), np.zeros(0), 16, 8, 8, 0.05, 0.25)


def test_model_config_rejects_invalid_budgets():
    for kw in (dict(n_tokens_scale2=0), dict(r1=0.0), dict(d_model=65), dict(min_wavelength=60.0)):
        with pytest.raises(ValueError):
            ModelConfig(**{**TINY.to_dict(), **kw})
    with pytest.raises(ValueError):
        CFD2vecNet(TINY, [0.0] * 6, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])


# ----------------------------------------------------------------------------------------------- geometry frame
def test_shifted_physical_frame_gives_identical_canonical_inputs():
    from cfd2vec.solvers.base import footprint_stats, wall_geometry
    rng = np.random.default_rng(0)
    fc = rng.uniform(0, 3, (400, 3)); fc[:200, 2] = 0.0
    fn = np.tile([0, 0, 1.0], (400, 1)); fa = rng.uniform(0.5, 1.5, 400)
    gm = np.arange(400) < 200
    pts = rng.uniform(0, 3, (1000, 3))
    t = np.array([123.4, -56.7, 8.9])
    o1, H1 = footprint_stats(fc, fn, fa, gm, 0.0)
    o2, H2 = footprint_stats(fc + t, fn, fa, gm, 0.0 + t[2])
    d1, n1 = wall_geometry(pts, fc, fn); d2, n2 = wall_geometry(pts + t, fc + t, fn)
    np.testing.assert_allclose((pts + t) - o2, pts - o1, atol=1e-9)
    assert abs(H1 - H2) < 1e-9
    np.testing.assert_allclose(d1, d2, atol=1e-4); np.testing.assert_allclose(n1, n2, atol=1e-4)


def test_rope_spatial_scores_relative_conditioning_scores_origin_anchored():
    """Documents the implemented behaviour: spatial-spatial scores are translation invariant; scores against a
    conditioning token (position 0) change when the spatial token moves relative to the canonical origin."""
    r = Rope3D(48, 0.02, 50.0)
    q, k = torch.randn(1, 1, 1, 48), torch.randn(1, 1, 1, 48)
    x, y, t = torch.randn(1, 1, 3), torch.randn(1, 1, 3), torch.tensor([[[0.7, -0.3, 0.2]]])
    torch.testing.assert_close((r(q, x) * r(k, y)).sum(), (r(q, x + t) * r(k, y + t)).sum(), atol=1e-4, rtol=1e-4)
    zero = torch.zeros(1, 1, 3)
    s1 = (r(q, x) * r(k, zero)).sum(); s2 = (r(q, x + t) * r(k, zero)).sum()
    assert not torch.allclose(s1, s2, atol=1e-4)


# ----------------------------------------------------------------------------------------------- diagnostics
def test_diagnostics_run_and_ridge_recovers_a_linear_target():
    from cfd2vec.eval import diagnostics as D
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)); y = X @ np.array([1.0, -2.0, 0.5, 0, 0]) + 0.05 * rng.normal(size=60)
    assert D.ridge_loo(X, y)["r2"] > 0.95 and D.ridge_loo(X, rng.normal(size=60))["r2"] < 0.2
    nest = D.ridge_nested_loo(X, y, n_boot=200)
    assert nest["r2"] > 0.95 and nest["r2_ci95"][0] > 0.9
    # pure noise with many features: the nested estimate must not report skill
    Xn = rng.normal(size=(40, 200)); yn = rng.normal(size=40)
    assert D.ridge_nested_loo(Xn, yn, n_boot=200)["r2"] < 0.1
    # the inner criterion equals brute-force leave-one-out refits with fold-wise centring, picks an interior penalty,
    # and does not degenerate to the smallest penalty when features outnumber cases
    Xs = (X - X.mean(0)) / X.std(0); ys = y + 2.0 * rng.normal(size=60)
    sse, a = D._dual_ridge_select(Xs, ys, D.ALPHAS, return_sse=True)
    brute = 0.0
    for i in range(60):
        tr = np.arange(60) != i
        mu = Xs[tr].mean(0)
        w = np.linalg.solve((Xs[tr] - mu).T @ (Xs[tr] - mu) + a * np.eye(5), (Xs[tr] - mu).T @ (ys[tr] - ys[tr].mean()))
        brute += (ys[i] - ys[tr].mean() - (Xs[i] - mu) @ w) ** 2
    Xw = rng.normal(size=(40, 300)); yw = Xw[:, 0] + rng.normal(size=40)
    assert D._dual_ridge_select(Xw, yw, D.ALPHAS) > min(D.ALPHAS)
    assert abs(sse - brute) / brute < 1e-6 and a not in (min(D.ALPHAS), max(D.ALPHAS))
    m = net(2)
    cases = [synthetic_case(n=1500, seed=i) for i in range(2)]
    rows = D.memory_dependence(m, cases, n_points=200)
    assert {"own_rel_l2_U", "zeroed_rel_l2_U", "swapped_rel_l2_U"} <= set(rows[0])
    g = D.gradient_transport(m, tb(make_sample(cases[0], spec(fixed_ratio=0.6), np.random.default_rng(0))))
    assert g["head"] > 0 and g["pointnet"] > 0
