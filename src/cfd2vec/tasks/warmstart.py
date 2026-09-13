"""Super-fidelity warm start: predict fields on the production mesh (optionally from a low-fidelity prior), seed the
unchanged solver with them, and guard the run with a residual-checked fallback to the standard cold start.

This is the AERO warm-start method expressed on the shared CFD2vec representation:
  * prior field    a cheap solve of the same problem (coarse twin, potential flow, mapped field) enters as the prior;
                   the decoder is residual on it and starts exactly at the prior
  * seeding        velocity only by default: seeding k / epsilon from a model can create non-physical nu_t spikes;
                   U,p and U,p,k,epsilon are available with positivity clamps
  * safety         after `probe_iter` iterations the pressure residual is compared with the acceptance rule; a case
                   that does not accelerate is restored to its original initial fields and rerun cold, which bounds
                   the worst case to the cold run plus the probe
The solver, mesh, schemes and convergence criteria are never modified. The run / safety part needs only NumPy, so
it can execute next to the solver on a machine without the ML stack.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..schema import Case, CLOSURES, Conditioning
from ..solvers.base import eps_mixing_length

K_EPS_FAMILY = {"k-epsilon", "realizable-k-epsilon"}


@dataclass
class SeedPolicy:
    fields: tuple = ("U",)
    k_floor: float = 1e-6          # x U_ref^2
    eps_floor: float = 1e-8        # x U_ref^3 / L_ref
    eps_source: str = "auto"       # "model" | "mixing-length" | "auto" (model when the closure is k-epsilon family)


@dataclass
class SafetyPolicy:
    enabled: bool = True
    probe_iter: int = 25
    field: str = "p"
    max_ratio_to_first: float = 0.5        # accept if residual(probe) <= ratio * residual(first iteration)
    accept_below: float = 1e-2             # or if residual(probe) is already this small (normalised)
    reference_residual: Optional[float] = None   # cold-start residual at probe_iter from a comparable case, if known
    blowup_residual: float = 1.0           # reject if any normalised residual reaches this after `settle_iter`
    settle_iter: int = 5


def to_physical(pred: np.ndarray, case: Case, policy: SeedPolicy = SeedPolicy()) -> dict:
    """Normalised [U, Cp, k, eps] -> solver fields in SI (kinematic pressure, gauge p_ref from case.meta), with
    positivity clamps. Raises if any converted field is non-finite."""
    U_ref, L = case.U_ref, case.L_ref
    k = np.maximum(pred[:, 4] * U_ref ** 2, policy.k_floor * U_ref ** 2)
    use_model_eps = policy.eps_source == "model" or (policy.eps_source == "auto" and case.cond.closure in K_EPS_FAMILY)
    if use_model_eps:
        eps = pred[:, 5] * U_ref ** 3 / L
    else:
        eps = eps_mixing_length(k, case.wall_dist * L, L)
    eps = np.maximum(eps, policy.eps_floor * U_ref ** 3 / L)
    p_ref = float(case.meta.get("p_ref", 0.0))            # gauge recorded by the adapter that read the case
    out = dict(U=pred[:, 0:3] * U_ref, p=0.5 * pred[:, 3] * U_ref ** 2 + p_ref, k=k, epsilon=eps,
               omega=eps / (0.09 * k))
    bad = [n for n, v in out.items() if not np.isfinite(v).all()]
    if bad:
        raise FloatingPointError(f"non-finite solver fields after conversion: {bad}")
    return out


def attach_prior(case: Case, prior_case: Case, kind: str = "coarse-twin") -> Case:
    """Map a low-fidelity solution onto the case points by nearest neighbour (both in the same frame)."""
    from scipy.spatial import cKDTree
    j = cKDTree(prior_case.points).query(case.points, workers=-1)[1]
    case.prior = prior_case.fields[j].astype(np.float32)
    case.prior_presence = prior_case.presence.copy()
    case.cond.prior_kind = kind
    return case


def safety_check(report, policy: SafetyPolicy) -> tuple[bool, str]:
    r = report.residuals.get(policy.field, [])
    if report.diverged:
        return False, "diverged"
    if len(r) < 2:
        return False, f"no {policy.field} residuals"
    for name, v in report.residuals.items():
        tail = v[policy.settle_iter:]
        if any(not np.isfinite(x) for x in v) or (tail and max(tail) >= policy.blowup_residual):
            return False, f"{name} residual blow-up"
    at = r[min(policy.probe_iter, len(r)) - 1]
    if policy.reference_residual is not None:
        return (at <= policy.reference_residual), f"{policy.field} residual {at:.3g} vs cold reference {policy.reference_residual:.3g}"
    ok = at <= policy.max_ratio_to_first * r[0] or at <= policy.accept_below
    return ok, f"{policy.field} residual {at:.3g} vs first {r[0]:.3g}"


def warm_start(model, adapter, case_dir: str, U_ref: float, L_ref: Optional[float], cond: Conditioning,
               seed_policy: SeedPolicy = SeedPolicy(), safety: SafetyPolicy = SafetyPolicy(),
               prior_dir: Optional[str] = None, run: bool = False, env_setup: Optional[str] = None,
               command=("foamRun",), device: str = "cpu", n_ensemble: int = 1) -> dict:
    """Write CFD2vec initial fields into `case_dir` and optionally run the solver with the safety fallback."""
    from .predict import predict_case
    if run:
        existing = time_dirs(case_dir)
        if existing:                                   # checked before anything is written
            raise FileExistsError(f"{case_dir}: existing solution time directories {existing[:5]}")
    if hasattr(adapter, "check_writable_layout"):
        adapter.check_writable_layout(case_dir)
    case = adapter.read_case(case_dir, U_ref=U_ref, L_ref=L_ref, cond=cond)
    if prior_dir:
        pc = adapter.read_case(prior_dir, U_ref=U_ref, L_ref=case.L_ref, cond=Conditioning(**asdict(cond)),
                               with_fields=True, origin=np.asarray(case.meta["origin"]))
        attach_prior(case, pc)
    out = predict_case(model, case, use_prior=prior_dir is not None, device=device, n_ensemble=n_ensemble)
    phys = to_physical(out["fields"], case, seed_policy)
    written = adapter.write_initial(case_dir, {f: phys[f] for f in seed_policy.fields})
    rep = dict(case_dir=case_dir, n_cells=case.n, L_ref=case.L_ref, fields=list(seed_policy.fields), written=written,
               prior=bool(prior_dir), closure=cond.closure)
    if out.get("std") is not None:
        rep["U_std_mean"] = float(np.linalg.norm(out["std"][:, 0:3], axis=1).mean())
    if run:
        rep.update(run_with_safety(adapter, case_dir, safety, env_setup, command))
    json.dump(rep, open(os.path.join(case_dir, "cfd2vec_warmstart.json"), "w"), indent=1, default=str)
    return rep


def _is_time_dir(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def time_dirs(case_dir: str, start_time: str = "0") -> list:
    """Numeric time directories other than the start time, at the top level and inside processor*/."""
    out = []
    procs = sorted(os.path.join(case_dir, d) for d in os.listdir(case_dir) if d.startswith("processor"))
    for root in [case_dir] + procs:
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p) and _is_time_dir(d) and float(d) != float(start_time):
                out.append(os.path.relpath(p, case_dir))
    return sorted(out)


def run_with_safety(adapter, case_dir, safety: SafetyPolicy, env_setup=None, command=("foamRun",),
                    start_time: str = "0") -> dict:
    """Probe run to `probe_iter`, then either continue from the probe state or restore and run cold.
    Iterations and solver time of the probe are always charged to the arm.

    Data safety: the case must hold no solution time directories besides `start_time` (checked before anything
    runs), so every later time directory is output of this call and is the only thing a rejection deletes. The
    original controlDict is restored in a `finally` block, a non-zero solver exit status rejects the probe and is
    reported, and the cold rerun starts only from a verified backup of the original initial fields. If anything
    raises after the probe starts (probe, continuation of an accepted probe, restore, or cold rerun), every time
    directory this call created is removed, the original initial fields and controls are restored, and the error
    propagates; a failure of that recovery is attached to the error as a note."""
    existing = time_dirs(case_dir, start_time)
    if existing:
        raise FileExistsError(f"{case_dir}: existing solution time directories {existing[:5]}; the warm start "
                              "needs a case that starts from its initial fields (move them or use a copy)")
    if not safety.enabled:
        r = adapter.run(case_dir, command, env_setup=env_setup)
        return dict(accepted=True, iterations=r.iterations, converged=r.converged and not r.failed,
                    exec_time_s=r.exec_time_s, returncode=r.returncode)
    ctrl = os.path.join(case_dir, "system", "controlDict")
    saved = ctrl + ".cfd2vec"
    shutil.copy2(ctrl, saved)

    def discard_new_output():
        for rel in time_dirs(case_dir, start_time):        # created by this call only (checked empty above)
            shutil.rmtree(os.path.join(case_dir, rel))

    def recover():
        """Return the case to its state before this call: new time directories removed, original initial fields
        restored from the verified backup. A failure here is reported, not raised, so the original error surfaces."""
        problems = []
        try:
            discard_new_output()
        except Exception as e:  # noqa: BLE001
            problems.append(f"output cleanup: {e!r}")
        try:
            adapter.restore_initial(case_dir, start_time)
        except Exception as e:  # noqa: BLE001
            problems.append(f"initial-field restore: {e!r}")
        return problems

    try:
        r1 = adapter.run(case_dir, command, log_name="log.probe", env_setup=env_setup, end_time=safety.probe_iter)
        shutil.copy2(saved, ctrl)
        ok, why = safety_check(r1, safety)
        if r1.failed:
            ok, why = False, f"solver exit status {r1.returncode} during the probe"
        base = dict(probe_iterations=r1.iterations, probe_reason=why, probe_returncode=r1.returncode)
        if ok and r1.converged:
            return dict(base, accepted=True, iterations=r1.iterations, converged=True, exec_time_s=r1.exec_time_s,
                        returncode=r1.returncode)
        if ok:
            r2 = adapter.run(case_dir, command, log_name="log.run", env_setup=env_setup, continue_run=True)
            return dict(base, accepted=True, iterations=r1.iterations + r2.iterations,
                        converged=r2.converged and not r2.failed, exec_time_s=r1.exec_time_s + r2.exec_time_s,
                        returncode=r2.returncode)
        discard_new_output()
        backup = adapter.restore_initial(case_dir, start_time)
        r2 = adapter.run(case_dir, command, log_name="log.run", env_setup=env_setup)
        return dict(base, accepted=False, iterations=r1.iterations + r2.iterations,
                    converged=r2.converged and not r2.failed, exec_time_s=r1.exec_time_s + r2.exec_time_s,
                    fallback="cold start", restored_from=backup, returncode=r2.returncode)
    except Exception as e:
        # any exception in the probe, the continuation, the restore or the cold rerun: undo this call's changes
        problems = recover()
        if problems and hasattr(e, "add_note"):
            e.add_note("recovery incomplete: " + "; ".join(problems))
        raise
    finally:
        shutil.copy2(saved, ctrl)
        os.remove(saved)


def warmstart_cli(a):
    from ..api import CFD2vec
    from ..solvers import get_adapter
    m = CFD2vec.from_pretrained(a.ckpt)
    cond = Conditioning(closure=a.closure if a.closure in CLOSURES else "unknown", abl_alpha=a.alpha,
                        turb_intensity=a.turb_intensity, ground="stationary",
                        inflow_dir=tuple(float(x) for x in a.inflow_dir.split(",")))
    rep = warm_start(m.net, get_adapter(a.solver), a.case_dir, a.U_ref, a.L_ref if a.L_ref > 0 else None, cond,
                     SeedPolicy(fields=tuple(a.fields.split(","))), prior_dir=a.prior_dir, device=m.device)
    print(json.dumps(rep, indent=1, default=str))
