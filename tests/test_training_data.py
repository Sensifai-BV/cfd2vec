"""Training-data contracts: full-pool eligibility, keyed randomness, persistent workers, pool validation, resume."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cfd2vec.train.dataset import KeyedSampler, ShardDataset, load_pool, numpy_collate, pool_fingerprint

from conftest import synthetic_case
from test_model import TINY, spec_for


def test_every_case_is_eligible_whatever_the_batch_size():
    n = 1326
    for bs in (1, 2, 3, 4, 8):
        s = KeyedSampler(n, seed=0); s.set_epoch(0)
        keys = list(s)
        consumed = keys[:len(keys) // bs * bs]                   # drop_last
        assert sorted(k[0] for k in keys) == list(range(n))      # one epoch emits every case once
        assert n - len({k[0] for k in consumed}) == n % bs       # only the drop_last remainder is not consumed
    e0 = [k[0] for k in KeyedSampler(n, 0)]
    s = KeyedSampler(n, 0); s.set_epoch(1)
    assert [k[0] for k in s] != e0                               # a fresh permutation each epoch


def test_sampler_cursor_continues_the_same_key_stream():
    s = KeyedSampler(50, seed=3); s.set_epoch(4)
    full = list(s)
    s.set_epoch(4, start=17)
    assert list(s) == full[17:] and len(s) == 33


def test_weighted_sampler_shares_and_validation():
    groups = ["a"] * 10 + ["b"] * 90
    s = KeyedSampler(100, 0, groups=groups, weights={"a": 0.5, "b": 0.5})
    keys = list(s)
    cnt = pd.Series([groups[k[0]] for k in keys]).value_counts()
    assert cnt["a"] == 50 and cnt["b"] == 50
    a_counts = pd.Series([k[0] for k in keys if groups[k[0]] == "a"]).value_counts()
    assert a_counts.min() == a_counts.max() == 5                 # cycles evenly through the small group
    with pytest.raises(ValueError):
        KeyedSampler(100, 0, groups=groups, weights={"a": 1.0})


@pytest.fixture
def shards(tmp_path):
    paths = []
    for i in range(6):
        p = str(tmp_path / f"c{i}.npz"); synthetic_case(n=2500, seed=i).save(p); paths.append(p)
    return paths


def test_sample_keys_fix_the_sample(shards):
    ds = ShardDataset(shards, spec_for(TINY), seed=7); ds.numpy_out = True
    a, b, c = ds[(2, 0, 5)], ds[(2, 0, 5)], ds[(2, 1, 5)]
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert not np.array_equal(a["ctx_vis"], c["ctx_vis"]) or not np.array_equal(a["q_pos"], c["q_pos"])
    np.testing.assert_array_equal(ds[3]["q_pos"], ds[(3, 0, 0)]["q_pos"])      # integer index = fixed key


def test_persistent_workers_see_new_epochs_and_match_main_process(shards):
    """Keys travel from the main-process sampler to the workers, so persistent workers produce epoch-dependent
    samples identical to a main-process evaluation of the same key, for any worker count."""
    from torch.utils.data import DataLoader
    ds = ShardDataset(shards, spec_for(TINY), seed=1); ds.numpy_out = True
    ref = {}
    for nw in (0, 2):
        s = KeyedSampler(len(shards), seed=1)
        try:
            dl = DataLoader(ds, batch_size=2, sampler=s, num_workers=nw, persistent_workers=nw > 0,
                            collate_fn=numpy_collate, drop_last=True)
            got = {}
            for ep in (0, 1):
                s.set_epoch(ep)
                for b in dl:
                    for j, key in enumerate(b["sample_key"]):
                        got[tuple(int(x) for x in key)] = b["ctx_vis"][j].copy()
        except (RuntimeError, PermissionError, OSError) as e:        # restricted sandboxes block worker processes
            pytest.skip(f"worker processes unavailable: {e}")
        ref[nw] = got
    assert ref[0].keys() == ref[2].keys()
    for key, v in ref[0].items():
        np.testing.assert_array_equal(v, ref[2][key])
        np.testing.assert_array_equal(v, ds.sample(key)["ctx_vis"])
    e0 = {k[0]: v for k, v in ref[0].items() if k[1] == 0}
    e1 = {k[0]: v for k, v in ref[0].items() if k[1] == 1}
    common = set(e0) & set(e1)
    assert common and any(not np.array_equal(e0[i], e1[i]) for i in common)


# ----------------------------------------------------------------------------------------------- pool
def manifest(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False); return str(path)


def row(cid, src, split, **kw):
    return dict(case_id=cid, source=src, split=split, shard=f"/x/{cid}.npz", **kw)


def test_two_source_pool_membership_and_rejections(tmp_path):
    a = manifest(tmp_path / "a.parquet", [row(f"u{i}", "aero_urban", "train" if i < 8 else "val") for i in range(10)])
    b = manifest(tmp_path / "b.parquet", [row(f"c{i}", "ahmedml", "train" if i < 4 else "test", validated=True,
                                              family=f"f{i}") for i in range(6)])
    pool = load_pool([a, b], sources=["aero_urban", "ahmedml"])
    assert len(pool) == 16 and set(pool.source) == {"aero_urban", "ahmedml"}
    assert pool_fingerprint(pool) == pool_fingerprint(load_pool([b, a]))
    assert set(load_pool([a, b], sources=["aero_urban"]).source) == {"aero_urban"}
    with pytest.raises(ValueError, match="not found in any manifest"):
        load_pool([a], sources=["aero_urban", "ahmedml"])
    with pytest.raises(FileNotFoundError):
        load_pool([a, str(tmp_path / "missing.parquet")])
    with pytest.raises(ValueError, match="held-out"):
        load_pool([a, b], exclude_sources=["ahmedml"])
    dup = manifest(tmp_path / "d.parquet", [row("u1", "aero_urban", "test")])
    with pytest.raises(ValueError, match="more than once"):
        load_pool([a, dup])
    fam = manifest(tmp_path / "f.parquet", [row("c0", "ahmedml", "train", validated=True, family="F"),
                                            row("c1", "ahmedml", "test", validated=True, family="F")])
    with pytest.raises(ValueError, match="families"):
        load_pool([fam])
    unval = manifest(tmp_path / "v.parquet", [row("w0", "windsorml", "train")])
    with pytest.raises(ValueError, match="validated"):
        load_pool([unval])


# ----------------------------------------------------------------------------------------------- pretraining audit
def write_run_config(tmp_path, shards, name="cfg.yaml", **over):
    man = manifest(tmp_path / "manifest_synthetic.parquet",
                   [dict(case_id=f"s{i}", source="synthetic", split="train" if i < 5 else "val", shard=p, tier="T",
                         validated=True) for i, p in enumerate(shards)])
    mc = tmp_path / "model.yaml"
    mc.write_text(yaml.safe_dump(dict(TINY.to_dict(), name="tiny")))
    st = tmp_path / "stats.json"; st.write_text(json.dumps(dict(mean=[0.0] * 6, std=[1.0] * 6)))
    y = dict(model=str(mc), manifests=[man], sources=["synthetic"], channel_stats=str(st), batch_size=2,
             max_steps=6, optim=dict(lr=1e-3, warmup_steps=100), eval_every=1000, eval_cases=1,
             val_recon_cases=1, num_workers=0, log_every=1, seed=0)
    y.update(over)
    p = tmp_path / name; p.write_text(yaml.safe_dump(y)); return str(p)


def test_pretrain_records_provenance_coverage_and_resumes_exactly(tmp_path, shards):
    """A run stopped at step 3 and resumed to step 6 ends with the same weights as an uninterrupted 6-step run
    (CPU; warm-up longer than the run keeps the learning rate independent of max_steps)."""
    from cfd2vec.train.pretrain import pretrain
    torch.set_num_threads(1)
    cfg = write_run_config(tmp_path, shards)
    ra, rb = str(tmp_path / "A"), str(tmp_path / "B")
    pretrain(cfg, ra, device="cpu")
    pretrain(cfg, rb, device="cpu", max_steps=3)
    pretrain(cfg, rb, device="cpu", max_steps=6, resume=True)
    a = torch.load(os.path.join(ra, "last.pt"), weights_only=False)
    b = torch.load(os.path.join(rb, "last.pt"), weights_only=False)
    assert a["step"] == b["step"] == 6
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k], rtol=0, atol=0)
    run = json.load(open(os.path.join(ra, "run.json")))
    prov = run["provenance"]
    assert len(prov["source"]["source_sha256"]) == 64 and prov["pool_sha256"] and prov["manifests"]
    assert run["epoch_length"] == 5 and run["keys_dropped_per_epoch"] == 1 and prov["input_schema"] == 2
    cov = json.load(open(os.path.join(ra, "coverage.json")))
    assert cov["draws"] == 12 and cov["n_distinct"] == 5 and cov["by_source"]["synthetic"]["n"] == 5
    assert a["sampler"]["epoch"] == b["sampler"]["epoch"] and a["sampler"]["cursor"] == b["sampler"]["cursor"]
    assert len(pd.read_csv(os.path.join(ra, "pool.csv"))) == 6


def test_resume_refuses_a_changed_pool_or_schema(tmp_path, shards):
    from cfd2vec.train.pretrain import pretrain
    cfg = write_run_config(tmp_path, shards, max_steps=2)
    out = str(tmp_path / "R")
    pretrain(cfg, out, device="cpu")
    cfg2 = write_run_config(tmp_path, shards[:5] + [shards[5]], name="cfg2.yaml", max_steps=4)
    man = pd.read_parquet(tmp_path / "manifest_synthetic.parquet")
    man.loc[0, "split"] = "val"; man.to_parquet(tmp_path / "manifest_synthetic.parquet", index=False)
    with pytest.raises(ValueError, match="pool changed"):
        pretrain(cfg2, out, device="cpu", resume=True)
