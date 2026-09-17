"""External-source ingestion contract and packaged protocol resolution."""
import glob
import json
import os

import numpy as np
import pytest

from cfd2vec.sources.caeml import (SourceNotValidated, from_canonical, k_from_stress, load_manifest, to_canonical,
                                   validate_manifest)

ROOT = os.path.join(os.path.dirname(__file__), "..")

GOOD = dict(source="ahmedml", solid_mask="none", stress_order="vtk", pressure={"convention": "kinematic"},
            inflow_dir=[1.0, 0.0, 0.0], L_ref={"rule": "x-extent"}, origin={"rule": "bbox-centre-ground"},
            ground_z=0.0, licence="CC-BY-SA-4.0", verified_run="run_1 checked against published Cd")


def test_shipped_manifest_templates_fail_closed():
    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "source_manifests", "*.yaml")))
    assert len(paths) == 5
    for p in paths:
        with pytest.raises(SourceNotValidated):
            load_manifest(p)


def test_manifest_validation_rules():
    assert validate_manifest(dict(GOOD), "ahmedml")
    for k in GOOD:
        m = dict(GOOD); del m[k]
        with pytest.raises(SourceNotValidated, match=k):
            validate_manifest(m)
    for k, v in (("stress_order", "xyz"), ("inflow_dir", [0, 0, 0]), ("pressure", {"convention": "static"}),
                 ("solid_mask", {"field": "solid"}), ("origin", {"rule": "somewhere"}), ("licence", "VERIFY")):
        with pytest.raises(SourceNotValidated):
            validate_manifest(dict(GOOD, **{k: v}))
    with pytest.raises(SourceNotValidated, match="not 'drivaerml'"):
        validate_manifest(dict(GOOD), "drivaerml")
    with pytest.raises(SourceNotValidated, match="chord"):
        validate_manifest(dict(GOOD, source="hiliftaeroml"))
    assert validate_manifest(dict(GOOD, source="hiliftaeroml", L_ref={"value": 0.35}))


@pytest.mark.parametrize("pressure", [{"convention": "kinematic"},
                                      {"convention": "static", "rho": 1.18, "p_ref": 101325.0}])
def test_canonical_conversion_round_trips(pressure):
    rng = np.random.default_rng(0)
    U, p, k = rng.normal(size=(50, 3)) * 30, rng.normal(size=50) * 200 + (101325 if "rho" in pressure else 0), \
        rng.random(50) * 5
    f = to_canonical(U, p, k, 38.9, pressure)
    U2, p2, k2 = from_canonical(f, 38.9, pressure)
    np.testing.assert_allclose(U2, U, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(p2, p, rtol=1e-6, atol=5e-2)
    np.testing.assert_allclose(k2, k, rtol=1e-5, atol=1e-6)


def test_stress_orders():
    R = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    assert k_from_stress(R, "vtk")[0] == 3.0          # XX YY ZZ = 1 2 3
    assert k_from_stress(R, "foam")[0] == 5.5         # XX YY ZZ = 1 4 6
    with pytest.raises(ValueError):
        k_from_stress(R, "none")


# ----------------------------------------------------------------------------------------------- packaging
def test_packaged_protocols_match_the_frozen_config_copies():
    from cfd2vec.api import DEFAULT_PROTOCOL, protocol_path
    import cfd2vec
    assert os.path.dirname(DEFAULT_PROTOCOL).startswith(os.path.dirname(cfd2vec.__file__))
    for name in ("finetune_protocol.yaml", "warmstart_protocol.yaml"):
        assert open(protocol_path(name), "rb").read() == open(os.path.join(ROOT, "configs", name), "rb").read()


def test_default_finetune_records_the_resolved_protocol(tmp_path):
    from cfd2vec.api import CFD2vec
    from cfd2vec.model.network import CFD2vecNet
    from conftest import synthetic_case
    from test_model import TINY
    paths = []
    for i in range(2):
        p = str(tmp_path / f"c{i}.npz"); synthetic_case(n=1500, seed=i).save(p); paths.append(p)
    m = CFD2vec(CFD2vecNet(TINY), "cpu"); m.stats = dict(mean=[0.0] * 6, std=[1.0] * 6)
    log = str(tmp_path / "ft.json")
    m.finetune(paths, max_epochs=1, log_path=log)
    rec = json.load(open(log))
    assert rec["protocol_version"] == 2 and len(rec["protocol_sha256"]) == 64 and rec["input_schema"] == 2
