"""Inference service: wire codec against protobuf, and the served prediction against the library path."""
import copy
import io
import json
import os
import tarfile

import numpy as np
import pytest

from cfd2vec.serve import wire

BENCH = os.environ.get("CFD2VEC_TEST_CASE", "/Users/javad/External/Datasets/cfd/aero_bench/B1")
CKPT = os.environ.get("CFD2VEC_TEST_CKPT", "runs/S_urban_v2_masked/last.pt")


@pytest.mark.parametrize("n", [0, 1, 127, 128, 300, 1 << 20])
def test_chunk_matches_protobuf(n):
    wrappers = pytest.importorskip("google.protobuf.wrappers_pb2")
    data = os.urandom(n)
    ref = wrappers.BytesValue(value=data).SerializeToString()      # same layout: bytes field 1
    assert wire.encode_chunk(data) == ref
    assert wire.decode_chunk(ref) == data


def test_frame_roundtrip():
    h = dict(a=1, b=[1, 2], c="x")
    buf = wire.frame(h, b"payload" * 1000)
    parts = list(wire.split(buf, 999))
    h2, p2 = wire.unframe(wire.join(wire.decode_chunk(wire.encode_chunk(p)) for p in parts))
    assert h2 == h and p2 == b"payload" * 1000


def test_untar_rejects_escape(tmp_path):
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as t:
        info = tarfile.TarInfo("../evil"); info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        wire.untar(bio.getvalue(), str(tmp_path))


@pytest.mark.skipif(not (os.path.isdir(os.path.join(BENCH, "coarse")) and os.path.exists(CKPT)),
                    reason="benchmark case or checkpoint not available")
def test_served_fields_match_library(tmp_path):
    from foamlib import FoamFieldFile
    from cfd2vec.schema import Conditioning
    from cfd2vec.serve.server import Predictor
    from cfd2vec.tasks.warmstart import SeedPolicy, to_physical
    import torch
    torch.set_num_threads(2)
    case_dir = os.path.join(BENCH, "coarse")                        # small mesh keeps the CPU test short
    prm = json.load(open(os.path.join(BENCH, "params.json")))["params"]
    cond = dict(closure="k-epsilon", abl_alpha=prm["alpha"], turb_intensity=prm.get("I"), ground="stationary",
                rotation_ok=True)
    hdr = dict(case_id="t", U_ref=prm["Uref"], nu=prm["nu"], L_ref=None, cond=cond, fields=["U", "p", "k", "epsilon"],
               use_prior=False)
    pred = Predictor(CKPT, device="cpu")
    resp = pred.predict_frame(wire.frame(hdr, wire.tar_paths(wire.case_entries(case_dir, "0", "case"))), 0.0)
    h, body = wire.unframe(resp)
    assert h["ok"], h
    wire.untar(body, str(tmp_path))
    # library path on the same case
    lib = pred.ad.read_case(case_dir, U_ref=prm["Uref"], L_ref=None, cond=Conditioning(**copy.deepcopy(cond)))
    lib.cond.log10_re = float(np.log10(lib.U_ref * lib.L_ref / prm["nu"]))
    assert abs(lib.cond.log10_re - h["log10_re"]) < 1e-12
    ref = to_physical(pred.m.predict(lib)["fields"], lib, SeedPolicy())
    for f in hdr["fields"]:
        got = np.asarray(FoamFieldFile(str(tmp_path / "0" / f)).internal_field)
        np.testing.assert_allclose(got, ref[f], rtol=1e-5, atol=1e-6 * float(np.abs(ref[f]).max()))
