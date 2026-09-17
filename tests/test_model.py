import numpy as np
import torch

from cfd2vec.model.geometry import build_tokens, coverage, farthest_point_sampling
from cfd2vec.model.masking import point_mask
from cfd2vec.model.network import CFD2vecNet, ModelConfig, Rope3D, count_parameters
from cfd2vec.train.dataset import SampleSpec, make_sample, to_torch

from conftest import synthetic_case

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, n_tokens_scale1=64, n_tokens_scale2=32, k_neighbors=16,
                   pointnet_hidden=[32, 64], n_context=2048, n_query=256)


def spec_for(cfg, **kw):
    return SampleSpec(n_context=cfg.n_context, n_query=cfg.n_query, n_tokens_scale1=cfg.n_tokens_scale1,
                      n_tokens_scale2=cfg.n_tokens_scale2, k_neighbors=cfg.k_neighbors, **kw)


def batch(cfg, n=2, **kw):
    rng = np.random.default_rng(0)
    ss = [to_torch(make_sample(synthetic_case(seed=i), spec_for(cfg, **kw), rng)) for i in range(n)]
    return {k: torch.stack([s[k] for s in ss]) for k in ss[0]}


def test_parameter_count_S():
    import os
    cfg = ModelConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "configs", "model_S.yaml"))
    n = count_parameters(CFD2vecNet(cfg))
    assert 25e6 < n < 40e6, n


def test_forward_shapes_and_loss():
    b = batch(TINY)
    m = CFD2vecNet(TINY)
    pred, glob = m(b)
    assert pred.shape == (2, TINY.n_query, 6) and glob.shape == (2, 2 * TINY.d_model)
    loss, per = m.loss(pred, b)
    loss.backward()
    assert torch.isfinite(loss) and per.shape == (6,)


def test_zero_init_head_returns_prior():
    b = batch(TINY, fixed_ratio=1.0, use_prior=True)
    m = CFD2vecNet(TINY).eval()
    with torch.no_grad():
        pred, _ = m(b)
    np.testing.assert_allclose(pred.numpy(), m.standardise(b["q_prior"]).numpy(), atol=1e-5)


def test_fps_coverage_and_weighting():
    c = synthetic_case(n=4000)
    tok = build_tokens(c.points, c.wall_dist, 256, 128, 32, 0.05, 0.25)
    assert coverage(c.n, tok["nb_idx"]) > 0.8
    near = c.wall_dist[tok["centre_idx"][tok["scale"] == 0]].mean()
    unif = c.wall_dist[tok["centre_idx"][tok["scale"] == 1]].mean()
    assert near < unif                      # scale index 0 (wall-weighted) concentrates at walls
    idx = farthest_point_sampling(torch.from_numpy(c.points), 64)
    assert idx.unique().numel() == 64


def test_mask_ratio_and_modes():
    c = synthetic_case(n=4000); rng = np.random.default_rng(0)
    cen = np.arange(0, 4000, 40)
    for mode in ("tokens", "near_wall", "wake", "boxes"):
        m = point_mask(c.points, c.wall_dist, c.streamwise(), cen, 0.6, mode, rng)
        assert abs(m.mean() - 0.6) < 0.05, mode
    assert point_mask(c.points, c.wall_dist, c.streamwise(), cen, 1.0, "tokens", rng).all()


def test_hidden_fields_do_not_leak():
    b = batch(TINY, fixed_ratio=1.0)
    assert float(b["ctx_field"].abs().sum()) == 0.0 and not b["ctx_vis"].any()


def test_rope_relative():
    r = Rope3D(48, 0.02, 50.0)
    q, k = torch.randn(1, 1, 2, 48), torch.randn(1, 1, 2, 48)
    p = torch.randn(1, 2, 3); t = torch.randn(1, 1, 3)
    s1 = (r(q, p) * r(k, p.flip(1))).sum(-1)
    s2 = (r(q, p + t) * r(k, p.flip(1) + t)).sum(-1)
    torch.testing.assert_close(s1, s2, atol=1e-4, rtol=1e-4)


def test_decoder_on_arbitrary_query_mesh():
    rng = np.random.default_rng(1)
    ctx, qry = synthetic_case(n=3000, seed=0), synthetic_case(n=500, seed=7)
    s = to_torch(make_sample(ctx, spec_for(TINY, fixed_ratio=1.0, augment=False), rng, query_case=qry))
    b = {k: v[None] for k, v in s.items()}
    m = CFD2vecNet(TINY).eval()
    with torch.no_grad():
        mem, pos, _ = m.encode(b)
        out = m.decode(mem, pos, b, chunk=128)
    assert out.shape == (1, 500, 6)


def test_context_padding_is_inert_and_batches():
    from cfd2vec.train.dataset import pad_context, numpy_collate
    small, big = synthetic_case(n=1500, seed=3), synthetic_case(n=3000, seed=4)
    spec = spec_for(TINY, fixed_ratio=1.0, augment=False)          # n_context 2048 > 1500 points
    s1 = make_sample(small, spec, np.random.default_rng(0))
    s2 = make_sample(big, spec, np.random.default_rng(0))
    assert s1["ctx_pos"].shape[0] == s2["ctx_pos"].shape[0] == TINY.n_context
    numpy_collate([s1, s2])                                         # stacks without error
    # same sample without padding gives the same prediction
    raw = make_sample(small, spec, np.random.default_rng(0))
    unpadded = {k: (v[:1500] if k.startswith("ctx_") else v) for k, v in raw.items()}
    m = CFD2vecNet(TINY).eval()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    with torch.no_grad():
        a = m({k: v[None] for k, v in to_torch(s1).items()})[0]
        b = m({k: v[None] for k, v in to_torch(unpadded).items()})[0]
    torch.testing.assert_close(a, b)
