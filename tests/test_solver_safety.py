"""X5 convergence scoring and warm-start safety orchestration, with fixtures and a fake solver (no OpenFOAM)."""
import os

import numpy as np
import pytest

from cfd2vec.solvers.base import RunReport
from cfd2vec.solvers.openfoam import BACKUP_SUFFIX, OpenFOAMAdapter, make_backup, verify_backup
from cfd2vec.tasks.benchmark import InvalidEvidence, evaluate_convergence, iterations_to_criterion, score_arm
from cfd2vec.tasks.warmstart import SafetyPolicy, run_with_safety

FIELDS = ["p", "Ux"]
PROTO = dict(convergence=dict(residual_fields=FIELDS, residual_tol=1e-3, integral_quantities=["drag"],
                              stationarity_window=20, stationarity_rel=0.01))


def log_text(n, start=0, fields=FIELDS, bad_at=None):
    lines = []
    for i in range(start + 1, start + n + 1):
        lines.append(f"Time = {i}")
        for f in fields:
            r = "nan" if bad_at == i else f"{max(10 ** (-i / 20), 1e-6):.6g}"
            lines.append(f"smoothSolver:  Solving for {f}, Initial residual = {r}, Final residual = 1e-9, No Iterations 2")
            lines.append(f"GAMG:  Solving for {f}, Initial residual = 0.5, Final residual = 1e-9, No Iterations 1")
    lines.append("ExecutionTime = 2.0 s  ClockTime = 2 s")
    return "\n".join(lines) + "\n"


def forces(path, its, start_dir="0"):
    d = os.path.join(path, "postProcessing", "cfd2vecForces", start_dir); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "forces.dat"), "w") as f:
        f.write("# Time total_x total_y total_z\n")
        for i in its:
            f.write(f"{i} ({1.0 + 0.5 * np.exp(-i / 10):.8g} 0 0) (0 0 0) (0 0 0)\n")


def arm(tmp_path, n=150, log_kw=None, force_its=None):
    d = tmp_path / "arm"; d.mkdir(exist_ok=True)
    (d / "log.run").write_text(log_text(n, **(log_kw or {})))
    if force_its is not None:
        forces(str(d), force_its)
    return str(d)


# ----------------------------------------------------------------------------------------------- scoring
def test_parse_log_keeps_time_ids_and_first_solve_residual():
    r = OpenFOAMAdapter.parse_log(log_text(3, start=100))
    assert r.residual_iters["p"] == [101.0, 102.0, 103.0]
    assert r.residuals["p"][0] == pytest.approx(10 ** (-101 / 20))          # not the second (0.5) solve


def test_missing_residual_field_is_invalid_not_converged():
    # the reviewer's reproduction: only small p residuals, required [p, Ux], no monitors
    with pytest.raises(InvalidEvidence):
        iterations_to_criterion({"p": [1e-4] * 4}, {}, ["p", "Ux"], 1e-3, 0, 0.01)
    e = evaluate_convergence({"p": (np.arange(1, 5), [1e-4] * 4)}, {}, ["p", "Ux"], [], 1e-3, 0, 0.01)
    assert not e["valid"] and not e["converged"] and "Ux" in e["reason"]
    with pytest.raises(ValueError):
        evaluate_convergence({}, {}, [], [], 1e-3, 0, 0.01)


def test_valid_arm_converges_at_the_aligned_iteration(tmp_path):
    s = score_arm(arm(tmp_path, force_its=range(1, 151)), str(tmp_path / "arm" / "log.run"), PROTO)
    assert s["status"] == "converged" and s["admissible"]
    i = s["iterations_to_criterion"]
    assert i >= 60                                                   # residuals reach 1e-3 at iteration 60
    drag = 1.0 + 0.5 * np.exp(-np.arange(1, 151) / 10)
    assert np.max(np.abs(drag[i - 1:i + 20] - drag[i - 1])) <= 0.01 * drag[i - 1]
    assert np.max(np.abs(drag[i - 2:i + 19] - drag[i - 2])) > 0.01 * drag[i - 2] or i == 60


@pytest.mark.parametrize("case", ["no_monitor", "shifted", "gap", "nan", "diverged", "failed"])
def test_incomplete_or_bad_evidence_never_scores(tmp_path, case):
    kw = dict(force_its=range(1, 151))
    if case == "no_monitor":
        kw = dict(force_its=None)
    elif case == "shifted":
        kw = dict(force_its=range(1001, 1151))                       # monitor ids from a different run
    elif case == "gap":
        kw = dict(force_its=[i for i in range(1, 151) if i % 7])     # sparse output: no gap-free window
    elif case == "nan":
        kw["log_kw"] = dict(bad_at=80)
    d = arm(tmp_path, **kw)
    if case == "diverged":
        with open(os.path.join(d, "log.run"), "a") as f:
            f.write("--> FOAM FATAL ERROR\n")
    s = score_arm(d, os.path.join(d, "log.run"), PROTO, returncode=1 if case == "failed" else 0)
    assert s["iterations_to_criterion"] is None and not s["admissible"], s
    want = {"no_monitor": "invalid", "shifted": "invalid", "gap": "invalid", "nan": "diverged",
            "diverged": "diverged", "failed": "failed"}[case]
    assert s["status"] == want, s


@pytest.mark.parametrize("record", ["100 (nan 1 0) (0 0 0) (0 0 0)", "100 (inf 1 0) (0 0 0) (0 0 0)",
                                    "100 (1.0x 1 0) (0 0 0) (0 0 0)", "100 (1 0)"])
def test_non_finite_or_malformed_monitor_record_is_invalid(tmp_path, record):
    """The reviewer's reproduction: a force record (nan 1 0) used to be read as (1, 0, 0) and scored converged."""
    d = arm(tmp_path, force_its=range(1, 151))
    p = os.path.join(d, "postProcessing", "cfd2vecForces", "0", "forces.dat")
    lines = open(p).read().splitlines()
    lines[100] = record
    open(p, "w").write("\n".join(lines) + "\n")
    s = score_arm(d, os.path.join(d, "log.run"), PROTO)
    assert s["status"] == "invalid" and not s["admissible"] and s["iterations_to_criterion"] is None, s


def test_probe_records_must_have_a_consistent_width(tmp_path):
    from cfd2vec.tasks.benchmark import MalformedMonitor, read_probes
    p = tmp_path / "U"; p.write_text("# Probe 0\n1 (3 4 0) (0 0 1)\n2 (6 8 0)\n")
    with pytest.raises(MalformedMonitor):
        read_probes(str(p))
    p.write_text("1 (3 4 0) (nan 0 1)\n")
    it, S = read_probes(str(p))
    assert np.isnan(S[0, 1])                                           # kept, so scoring can reject it


def test_restarted_monitor_files_are_joined(tmp_path):
    d = tmp_path / "arm"; d.mkdir()
    (d / "log.run").write_text(log_text(150))
    forces(str(d), range(1, 81), "0"); forces(str(d), range(80, 151), "79")
    s = score_arm(str(d), str(d / "log.run"), PROTO)
    assert s["status"] == "converged" and s["n_aligned"] == 150


# ----------------------------------------------------------------------------------------------- safety orchestration
class FakeSolver(OpenFOAMAdapter):
    """Writes a time directory like a solver; the probe outcome is scripted."""

    def __init__(self, probe="reject", fail_on=None):
        self.probe, self.fail_on, self.calls = probe, fail_on, []

    def set_control(self, case_dir, entries, env_setup=None):
        p = os.path.join(case_dir, "system", "controlDict")
        with open(p, "a") as f:
            f.write("".join(f"{k} {v};\n" for k, v in entries.items()))

    def run(self, case_dir, command=("foamRun",), log_name="log.run", env_setup=None, end_time=None,
            continue_run=False):
        self.calls.append(log_name)
        if end_time is not None:
            self.set_control(case_dir, {"endTime": end_time})
        t = str(end_time or 500)
        os.makedirs(os.path.join(case_dir, t), exist_ok=True)
        open(os.path.join(case_dir, t, "U"), "w").write("solver output")
        if self.fail_on == log_name:                   # fails after writing partial output
            raise RuntimeError("injected solver failure")
        if log_name == "log.probe":
            r = [1.0] + ([0.1] * 24 if self.probe == "accept" else [0.9] * 24)
            return RunReport(25, False, 1.0, residuals={"p": r}, returncode=3 if self.probe == "exit3" else 0)
        return RunReport(500, True, 10.0, residuals={"p": [1e-5]}, returncode=0)


def make_case(tmp_path, backup=True, extra=()):
    d = tmp_path / "case"
    for sub in ("0", "system", "constant"):
        (d / sub).mkdir(parents=True)
    (d / "0" / "U").write_text("original initial U")
    (d / "system" / "controlDict").write_text("endTime 1000;\n")
    (d / "constant" / "keep").write_text("mesh data")
    for t in extra:
        (d / t).mkdir(); (d / t / "U").write_text(f"existing solution {t}")
    if backup:
        make_backup(str(d / "0"), str(d / f"0{BACKUP_SUFFIX}"))
    (d / "0" / "U").write_text("CFD2vec predicted U")                    # what write_initial would leave
    return d


def test_existing_solution_times_refuse_before_any_change(tmp_path):
    d = make_case(tmp_path, extra=("100", "1000"))
    ad = FakeSolver()
    with pytest.raises(FileExistsError):
        run_with_safety(ad, str(d), SafetyPolicy())
    assert ad.calls == [] and (d / "100" / "U").read_text() == "existing solution 100"
    assert (d / "system" / "controlDict").read_text() == "endTime 1000;\n"


def test_rejected_probe_removes_only_its_output_and_runs_cold_from_backup(tmp_path):
    d = make_case(tmp_path)
    rep = run_with_safety(FakeSolver("reject"), str(d), SafetyPolicy())
    assert not rep["accepted"] and rep["fallback"] == "cold start"
    assert not (d / "25").exists() and (d / "500").exists()               # probe output gone, cold run output kept
    assert (d / "0" / "U").read_text() == "original initial U"
    assert (d / "system" / "controlDict").read_text() == "endTime 1000;\n"
    assert (d / "constant" / "keep").read_text() == "mesh data"
    assert not (d / "system" / "controlDict.cfd2vec").exists()


def test_probe_exit_status_rejects(tmp_path):
    rep = run_with_safety(FakeSolver("exit3"), str(make_case(tmp_path)), SafetyPolicy())
    assert not rep["accepted"] and "exit status 3" in rep["probe_reason"]


def test_accepted_probe_continues_and_restores_controls(tmp_path):
    d = make_case(tmp_path)
    rep = run_with_safety(FakeSolver("accept"), str(d), SafetyPolicy())
    assert rep["accepted"] and (d / "0" / "U").read_text() == "CFD2vec predicted U"
    assert (d / "system" / "controlDict").read_text() == "endTime 1000;\n"


@pytest.mark.parametrize("probe,fail_on", [("reject", "log.probe"), ("reject", "log.run"), ("accept", "log.probe"),
                                           ("accept", "log.run")])
def test_solver_exception_restores_controls_and_original_data(tmp_path, probe, fail_on):
    """Covers the continuation of an accepted probe (the reviewer's reproduction left 25/ and the predicted fields)
    as well as the probe itself and the cold rerun after a rejection."""
    d = make_case(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        run_with_safety(FakeSolver(probe, fail_on=fail_on), str(d), SafetyPolicy())
    assert (d / "system" / "controlDict").read_text() == "endTime 1000;\n"
    assert (d / "constant" / "keep").read_text() == "mesh data"
    assert not (d / "25").exists() and not (d / "500").exists()
    assert (d / "0" / "U").read_text() == "original initial U"
    assert not (d / "system" / "controlDict.cfd2vec").exists()


def test_missing_backup_never_runs_cold_on_predicted_fields(tmp_path):
    d = make_case(tmp_path, backup=False)
    ad = FakeSolver("reject")
    with pytest.raises(FileNotFoundError, match="backup"):
        run_with_safety(ad, str(d), SafetyPolicy())
    assert ad.calls == ["log.probe"] and (d / "system" / "controlDict").read_text() == "endTime 1000;\n"


def test_backup_manifest_detects_modification(tmp_path):
    d = make_case(tmp_path)
    b = d / f"0{BACKUP_SUFFIX}"
    verify_backup(str(b))
    (b / "U").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        verify_backup(str(b))
    with pytest.raises(ValueError):
        OpenFOAMAdapter().restore_initial(str(d))


def test_decomposed_layout_is_rejected_before_mutation(tmp_path):
    d = make_case(tmp_path, backup=False)
    (d / "processor0" / "0").mkdir(parents=True)
    before = sorted(os.listdir(d))
    with pytest.raises(NotImplementedError, match="decomposed"):
        OpenFOAMAdapter().write_initial(str(d), {"U": np.zeros((4, 3))})
    assert sorted(os.listdir(d)) == before and (d / "0" / "U").read_text() == "CFD2vec predicted U"
