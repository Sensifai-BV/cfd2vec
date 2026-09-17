import json
import os

import numpy as np
import torch

from cfd2vec.api import CFD2vec
from cfd2vec.model.network import CFD2vecNet
from cfd2vec.solvers.base import RunReport, footprint_stats
from cfd2vec.solvers.openfoam import OpenFOAMAdapter
from cfd2vec.tasks.retrieval import EmbeddingIndex
from cfd2vec.tasks.warmstart import SafetyPolicy, SeedPolicy, attach_prior, safety_check, to_physical

from conftest import synthetic_case
from test_model import TINY

LOG = """sigFpe : Floating point exception trapping - not supported on this platform
Time = 1
smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.01, No Iterations 2
GAMG:  Solving for p, Initial residual = 1, Final residual = 0.001, No Iterations 5
Time = 2
smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 0.01, No Iterations 2
GAMG:  Solving for p, Initial residual = 0.2, Final residual = 0.001, No Iterations 5
ExecutionTime = 3.5 s  ClockTime = 4 s

SIMPLE solution converged in 2 iterations
"""


def test_parse_log():
    r = OpenFOAMAdapter.parse_log(LOG)
    assert r.iterations == 2 and r.converged and r.exec_time_s == 3.5
    assert r.residuals["p"] == [1.0, 0.2] and not r.diverged
    crash = OpenFOAMAdapter.parse_log(LOG + "\n#0  Foam::error::printStack(Foam::Ostream&)\nFloating point exception (core dumped)\n")
    assert crash.diverged


def test_safety_rules():
    good = RunReport(25, False, 1.0, residuals={"p": [1.0] + [0.1] * 24})
    bad = RunReport(25, False, 1.0, residuals={"p": [1.0] + [0.9] * 24})
    grow = RunReport(25, False, 1.0, residuals={"p": [1.0] + [0.1] * 23 + [50.0]})
    rising_start = RunReport(25, False, 1.0, residuals={"p": [0.02, 0.2, 0.3] + [0.005] * 22})   # healthy warm start
    assert safety_check(rising_start, SafetyPolicy(max_ratio_to_first=1.0))[0]
    assert safety_check(good, SafetyPolicy())[0]
    assert not safety_check(bad, SafetyPolicy())[0]
    assert not safety_check(grow, SafetyPolicy())[0]
    assert safety_check(bad, SafetyPolicy(reference_residual=0.95))[0]


def test_to_physical_clamps_and_units():
    c = synthetic_case(n=500); c.U_ref, c.L_ref = 2.0, 0.5
    f = c.fields.copy(); f[:10, 4] = -1.0; f[:10, 5] = -1.0
    ph = to_physical(f, c, SeedPolicy())
    np.testing.assert_allclose(ph["U"], f[:, 0:3] * 2.0)
    np.testing.assert_allclose(ph["p"], 0.5 * f[:, 3] * 4.0)
    assert (ph["k"] > 0).all() and (ph["epsilon"] > 0).all() and np.isfinite(ph["omega"]).all()
    c.cond.closure = "WMLES"                                # no eps from the model: mixing-length fallback
    assert (to_physical(f, c, SeedPolicy())["epsilon"] > 0).all()


def test_attach_prior_nearest():
    a, b = synthetic_case(n=800, seed=0, prior=False), synthetic_case(n=800, seed=0, prior=False)
    attach_prior(a, b)
    np.testing.assert_allclose(a.prior, b.fields)
    assert a.cond.prior_kind == "coarse-twin"


def test_footprint_of_unit_box():
    # upward faces of a unit box on the ground at x,y in [2,3]: roof at z=1
    fc = np.array([[2.5, 2.5, 1.0], [2.0, 2.5, 0.5], [2.5, 2.5, 0.0]]); fn = np.array([[0, 0, 1.0], [-1, 0, 0], [0, 0, 1.0]])
    fa = np.ones(3); gm = np.array([False, False, True])
    o, H = footprint_stats(fc, fn, fa, gm, 0.0)
    np.testing.assert_allclose(o, [2.5, 2.5, 0.0]); assert abs(H - 1.0) < 1e-9


def test_finetune_save_load_roundtrip(tmp_path):
    paths = []
    for i in range(3):
        p = str(tmp_path / f"c{i}.npz"); synthetic_case(n=2500, seed=i).save(p); paths.append(p)
    m = CFD2vec(CFD2vecNet(TINY), "cpu"); m.stats = dict(mean=[0.0] * 6, std=[1.0] * 6)
    proto = os.path.join(os.path.dirname(__file__), "..", "configs", "finetune_protocol.yaml")
    hist = m.finetune(paths, proto, max_epochs=2)
    assert len(hist) == 2 and np.isfinite(hist[-1]["loss"])          # N < 10: fixed schedule, no epoch-0 entry
    ck = str(tmp_path / "m.pt"); m.save(ck)
    m2 = CFD2vec.from_pretrained(ck, device="cpu")
    c = synthetic_case(n=1200, seed=9)
    a, b = m.predict(c)["fields"], m2.predict(c)["fields"]
    np.testing.assert_allclose(a, b, atol=1e-5)
    e = m2.encode(c)["embedding"]
    idx = EmbeddingIndex(["x", "y"], np.stack([e, e + 1.0]))
    assert idx.query(e, k=1)[0][0] == "x"


def test_convergence_criterion_needs_residuals_and_stationarity():
    from cfd2vec.tasks.benchmark import iterations_to_criterion
    n = 600
    res = {"p": list(np.geomspace(1, 1e-5, n)), "Ux": list(np.geomspace(1, 1e-5, n))}
    drag = 1.0 + 0.5 * np.exp(-np.arange(n) / 60.0)           # settles to within 1 % after ~ 60 ln(50) iterations
    i = iterations_to_criterion(res, {"drag": drag}, ["p", "Ux"], 1e-3, 200, 0.01)
    i_res = int(np.argmax(np.asarray(res["p"]) <= 1e-3)) + 1
    assert i is not None and i >= i_res
    assert np.max(np.abs(drag[i - 1:i + 200] - drag[i - 1])) <= 0.01 * drag[i - 1]
    assert np.max(np.abs(drag[i - 2:i + 199] - drag[i - 2])) > 0.01 * drag[i - 2] or i == i_res
    assert iterations_to_criterion(res, {"drag": drag}, ["p"], 1e-9, 200, 0.01) is None


def test_monitor_install_and_output_parsing(tmp_path):
    from cfd2vec.tasks.benchmark import (disable_residual_stop, install_monitors, monitor_functions, read_forces,
                                         read_probes)
    cd = tmp_path / "controlDict"; cd.write_text("application foamRun;\nendTime 10;\n")
    blk = monitor_functions(["buildings"], np.array([[0.1, 0.2, 0.3], [1, 2, 3]]))
    install_monitors(str(cd), blk); install_monitors(str(cd), blk)     # idempotent
    txt = cd.read_text()
    assert txt.count("cfd2vecForces") == 1 and "functions" in txt and "(buildings)" in txt
    fv = tmp_path / "fvSolution"; fv.write_text("SIMPLE\n{\n residualControl\n {\n  p 0.001;\n  U 0.001;\n }\n}\n")
    disable_residual_stop(str(fv))
    assert "1e-12" in fv.read_text() and "0.001" in (tmp_path / "fvSolution.orig").read_text()
    f = tmp_path / "forces.dat"; f.write_text("# Time total_x total_y total_z\n1 (1 2 3) (0 0 0) (1 2 3)\n2 1.5 2 3 0 0 0 1 1 1\n")
    it, F = read_forces(str(f)); assert list(it) == [1, 2] and F[1, 0] == 1.5
    p = tmp_path / "U"; p.write_text("# Probe 0\n1 (3 4 0) (0 0 1)\n2 (6 8 0) (0 0 2)\n")
    it, S = read_probes(str(p)); np.testing.assert_allclose(S, [[5, 1], [10, 2]])


def test_enable_potential_foam(tmp_path):
    from cfd2vec.tasks.benchmark import enable_potential_foam
    (tmp_path / "fvSolution").write_text("solvers\n{\n    p\n    {\n        solver GAMG;\n    }\n}\nSIMPLE\n{\n}\n")
    (tmp_path / "fvSchemes").write_text("divSchemes\n{\n    default none;\n}\n")
    enable_potential_foam(str(tmp_path)); enable_potential_foam(str(tmp_path))       # idempotent
    t = (tmp_path / "fvSolution").read_text()
    assert t.count("Phi") == 1 and t.count("potentialFlow") == 1 and "SIMPLE" in t
    assert (tmp_path / "fvSchemes").read_text().count("div(div(phi,U))") == 1
