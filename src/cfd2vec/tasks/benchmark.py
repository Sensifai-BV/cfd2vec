"""Solver warm-start benchmark (configs/warmstart_protocol.yaml): arm preparation, monitors, and the convergence
criterion combining residuals with integral-quantity stationarity. Needs only NumPy, so it runs next to the solver."""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np

FUNCTIONS_BEGIN = "// cfd2vec benchmark monitors: begin"
FUNCTIONS_END = "// cfd2vec benchmark monitors: end"


# ----------------------------------------------------------------------------------------------- monitors
def pedestrian_probes(points_m: np.ndarray, wall_dist_m: np.ndarray, z_target_m: float, dz_m: float,
                      ground_z_m: float = 0.0, n: int = 64, seed: int = 0,
                      xy_box: Optional[Sequence[float]] = None) -> np.ndarray:
    """Fixed probe locations: cell centres within dz of the target height whose nearest wall is the ground (street
    cells, not cells against a building face), sampled with a fixed seed (identical for every arm of a case)."""
    near = np.abs(points_m[:, 2] - z_target_m) <= dz_m
    street = wall_dist_m >= 0.95 * (points_m[:, 2] - ground_z_m)
    inside = np.ones(len(points_m), bool)
    if xy_box is not None:                          # (xmin, xmax, ymin, ymax): keep probes inside the built area
        x0, x1, y0, y1 = xy_box
        inside = (points_m[:, 0] >= x0) & (points_m[:, 0] <= x1) & (points_m[:, 1] >= y0) & (points_m[:, 1] <= y1)
    idx = np.flatnonzero(near & street & inside)
    if idx.size == 0:
        raise ValueError("no cells at pedestrian height")
    pick = np.random.default_rng(seed).choice(idx, size=min(n, idx.size), replace=False)
    return points_m[np.sort(pick)]


def monitor_functions(wall_patches: Sequence[str], probes_m: np.ndarray) -> str:
    """controlDict function objects for the benchmark. Both probe keywords are written because OpenFOAM versions
    differ in which one they read; the unused one is ignored."""
    pts = "\n".join(f"            ({x:.6g} {y:.6g} {z:.6g})" for x, y, z in probes_m)
    patches = " ".join(wall_patches)
    return f"""{FUNCTIONS_BEGIN}
    cfd2vecForces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         ({patches});
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
        pitchAxis       (0 1 0);
        writeControl    timeStep;
        writeInterval   1;
    }}
    cfd2vecProbes
    {{
        type            probes;
        libs            ("libsampling.so");
        fields          (U);
        probeLocations
        (
{pts}
        );
        points
        (
{pts}
        );
        writeControl    timeStep;
        writeInterval   1;
    }}
    {FUNCTIONS_END}"""


def install_monitors(control_dict: str, block: str) -> None:
    """Insert (or replace) the monitor block inside the `functions` dictionary of a controlDict."""
    txt = open(control_dict).read()
    txt = re.sub(re.escape(FUNCTIONS_BEGIN) + r".*?" + re.escape(FUNCTIONS_END), "", txt, flags=re.S)
    m = re.search(r"^functions\s*\{", txt, flags=re.M)
    if m:
        txt = txt[:m.end()] + "\n    " + block + "\n" + txt[m.end():]
    else:
        txt = txt.rstrip() + "\n\nfunctions\n{\n    " + block + "\n}\n"
    open(control_dict, "w").write(txt)


def disable_residual_stop(fv_solution: str, tol: float = 1e-12) -> None:
    """Replace every residualControl tolerance so the solver runs the full iteration budget; the criterion is
    evaluated afterwards. The original file is kept as fvSolution.orig."""
    txt = open(fv_solution).read()
    if not os.path.exists(fv_solution + ".orig"):
        open(fv_solution + ".orig", "w").write(txt)
    m = re.search(r"residualControl\s*\{(.*?)\}", txt, flags=re.S)
    if m:
        body = re.sub(r"([\w\"().|*]+)\s+[0-9.eE+-]+\s*;", lambda k: f"{k.group(1)} {tol:g};", m.group(1))
        txt = txt[:m.start(1)] + body + txt[m.end(1):]
    open(fv_solution, "w").write(txt)


# ----------------------------------------------------------------------------------------------- post-processing
class MalformedMonitor(ValueError):
    """A monitor file holds a record that cannot be parsed completely."""


def _numbers(line: str) -> list:
    """Every whitespace-separated token of a record (parentheses removed) as a float. `nan` and `inf` are kept as
    values; any other non-numeric token raises MalformedMonitor, so a record is never silently shortened."""
    out = []
    for tok in line.replace("(", " ").replace(")", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            raise MalformedMonitor(f"non-numeric token {tok!r} in record: {line.strip()[:120]}") from None
    return out


def _records(path: str):
    for n, line in enumerate(open(path), 1):
        if line.startswith("#") or not line.strip():
            continue
        yield n, _numbers(line)


def read_forces(path: str) -> tuple[np.ndarray, np.ndarray]:
    """forces.dat -> (iterations, total force (N,3)). The first three numbers after the time are the total force.
    Records with fewer than four numbers raise MalformedMonitor; non-finite values are returned as they are and
    make the evidence invalid when scored."""
    it, F = [], []
    for n, v in _records(path):
        if len(v) < 4:
            raise MalformedMonitor(f"{path}:{n}: {len(v)} numbers, a forces record needs at least 4")
        it.append(v[0]); F.append(v[1:4])
    return np.asarray(it, float), np.asarray(F, float).reshape(-1, 3)


def read_probes(path: str) -> tuple[np.ndarray, np.ndarray]:
    """probes U file -> (iterations, |U| per probe (N, n_probes)). Every record must hold the same number of
    3-vectors; otherwise MalformedMonitor is raised."""
    it, S, width = [], [], None
    for n, v in _records(path):
        if len(v) < 4 or (len(v) - 1) % 3:
            raise MalformedMonitor(f"{path}:{n}: {len(v)} numbers is not a time plus whole 3-vectors")
        if width is None:
            width = len(v)
        elif len(v) != width:
            raise MalformedMonitor(f"{path}:{n}: {len(v)} numbers, earlier records have {width}")
        it.append(v[0]); S.append(np.linalg.norm(np.asarray(v[1:]).reshape(-1, 3), axis=1))
    return np.asarray(it, float), np.asarray(S, float)


def find_outputs(case_dir: str, name: str, filename: str) -> list:
    """Every <case>/postProcessing/<name>/<start time>/<filename>, ordered by start time. A restarted run writes a
    new start-time directory, so all of them together cover the run."""
    root = os.path.join(case_dir, "postProcessing", name)
    if not os.path.isdir(root):
        return []
    out = []
    for d in os.listdir(root):
        try:
            t = float(d)
        except ValueError:
            continue
        p = os.path.join(root, d, filename)
        if os.path.exists(p):
            out.append((t, p))
    return [p for _, p in sorted(out)]


def find_output(case_dir: str, name: str, filename: str) -> Optional[str]:
    ps = find_outputs(case_dir, name, filename)
    return ps[0] if ps else None


def read_series(paths: Sequence[str], reader) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate monitor files in start-time order; for a repeated iteration id the later file wins."""
    rows = {}
    for p in paths:
        it, v = reader(p)
        for i, x in zip(it, v):
            rows[round(float(i), 9)] = x
    if not rows:
        return np.zeros(0), np.zeros(0)
    k = np.asarray(sorted(rows))
    return k, np.asarray([rows[i] for i in k])


class InvalidEvidence(ValueError):
    """The run's output cannot certify or refute convergence (missing, misaligned or non-finite data)."""


def evaluate_convergence(residuals: dict, quantities: dict, fields: Sequence[str], quantity_names: Sequence[str],
                         tol: float, window: int, rel: float) -> dict:
    """Apply the frozen criterion on iteration-aligned data.

    residuals:  field -> (iteration ids, initial residuals);  quantities: name -> (iteration ids, values).
    Every field in `fields` and every quantity in `quantity_names` is required; an empty `quantity_names` is an
    explicit residual-only criterion. Series are joined on iteration id. A candidate iteration i qualifies when every
    residual at i is <= tol and, for every quantity q, max over the next `window` iterations of |q_j - q_i| <=
    rel * |q_i|; the window must be gap-free in the joined data.
    Returns dict(valid, reason, converged, iteration_id, iteration, n_aligned). `iteration` counts from the first
    logged iteration (1-based)."""
    if not fields:
        raise ValueError("the criterion needs at least one residual field")
    out = dict(valid=False, reason="", converged=False, iteration_id=None, iteration=None, n_aligned=0)
    missing = [f for f in fields if f not in residuals or len(residuals[f][0]) == 0]
    missing += [q for q in quantity_names if q not in quantities or len(quantities[q][0]) == 0]
    if missing:
        out["reason"] = f"missing evidence: {missing}"
        return out
    series = [(f, residuals[f]) for f in fields] + [(q, quantities[q]) for q in quantity_names]
    aligned = {}
    for name, (it, v) in series:
        it = np.round(np.asarray(it, float), 9); v = np.asarray(v, float)
        if len(it) != len(v):
            out["reason"] = f"{name}: {len(it)} iteration ids for {len(v)} values"
            return out
        if len(np.unique(it)) != len(it):
            out["reason"] = f"{name}: repeated iteration ids"
            return out
        if not np.isfinite(v).all():
            out["reason"] = f"{name}: non-finite values"
            return out
        aligned[name] = dict(zip(it.tolist(), v.tolist()))
    common = sorted(set.intersection(*[set(a) for a in aligned.values()]))
    out["n_aligned"] = len(common)
    if not common:
        out["reason"] = "no iteration present in every series"
        return out
    ids = np.asarray(common)
    first_id = float(np.min(np.round(np.asarray(residuals[fields[0]][0], float), 9)))
    steps = np.diff(np.round(np.asarray(residuals[fields[0]][0], float), 9))
    step = float(np.median(steps)) if steps.size else 1.0
    R = np.stack([[aligned[f][i] for i in common] for f in fields])
    res_ok = (R <= tol).all(axis=0)
    Q = [np.asarray([aligned[q][i] for i in common]) for q in quantity_names]
    out.update(valid=True, reason="criterion not met within the aligned iterations")
    gap_blocked = 0
    for p in range(len(common) - window):
        if not res_ok[p]:
            continue
        if window and not np.isclose(ids[p + window] - ids[p], window * step):
            gap_blocked += 1                           # a gap inside the window: cannot certify stationarity
            continue
        if all(np.max(np.abs(q[p:p + window + 1] - q[p])) <= rel * max(abs(q[p]), 1e-30) for q in Q):
            out.update(converged=True, reason="criterion met", iteration_id=float(ids[p]),
                       iteration=int(round((ids[p] - first_id) / step)) + 1)
            return out
    if gap_blocked:
        out.update(valid=False, reason=f"gaps in the aligned data blocked {gap_blocked} candidate iterations")
    return out


def iterations_to_criterion(residuals: dict, quantities: dict, fields: Sequence[str], tol: float,
                            window: int, rel: float) -> Optional[int]:
    """Index-aligned form of `evaluate_convergence` for per-iteration lists (iteration ids 1..n). Every listed field
    and every quantity passed is required. Returns the first qualifying iteration (1-based), None if never met;
    raises InvalidEvidence on missing or non-finite data."""
    def ids(v):
        return np.arange(1, len(v) + 1, dtype=float)
    r = {f: (ids(v), v) for f, v in residuals.items()}
    q = {k: (ids(v), v) for k, v in quantities.items()}
    e = evaluate_convergence(r, q, fields, list(quantities), tol, window, rel)
    if not e["valid"]:
        raise InvalidEvidence(e["reason"])
    return e["iteration"] if e["converged"] else None


def score_arm(case_dir: str, log_path: str, protocol: dict, inflow_dir=(1.0, 0.0, 0.0),
              returncode: Optional[int] = None) -> dict:
    """Score one finished arm under the protocol. `status` is one of
        converged      criterion met on complete, aligned, finite evidence
        not_converged  complete evidence, criterion never met within the budget
        invalid        required residuals or monitors missing, misaligned or non-finite
        diverged       the log shows divergence or a crash
        failed         the solver exited with a non-zero status
    Only `converged` and `not_converged` are admissible in success summaries (`admissible`), and
    `iterations_to_criterion` is set only for `converged`."""
    from ..solvers.openfoam import OpenFOAMAdapter
    rep = OpenFOAMAdapter.parse_log(open(log_path).read())
    cv = protocol["convergence"]
    res = {f: (np.asarray(rep.residual_iters.get(f, []), float), np.asarray(v, float))
           for f, v in rep.residuals.items()}
    q, malformed = {}, None
    try:
        fps = find_outputs(case_dir, "cfd2vecForces", "forces.dat")
        if fps:
            it, F = read_series(fps, read_forces)
            if len(it):
                q["drag"] = (it, F @ np.asarray(inflow_dir, float))
        pps = find_outputs(case_dir, "cfd2vecProbes", "U")
        if pps:
            it, S = read_series(pps, read_probes)
            if len(it):
                q["pedestrian_speed"] = (it, S.mean(axis=1))
    except MalformedMonitor as e:
        malformed = str(e)
    names = list(cv["integral_quantities"])
    crit = evaluate_convergence(res, q, cv["residual_fields"], names, cv["residual_tol"],
                                cv["stationarity_window"], cv["stationarity_rel"])
    if malformed:
        crit.update(valid=False, converged=False, iteration=None, iteration_id=None,
                    reason=f"malformed monitor output: {malformed}")
    res_only = evaluate_convergence(res, {}, cv["residual_fields"], [], cv["residual_tol"], 0, 1.0)
    if returncode not in (None, 0):
        status = "failed"
    elif rep.diverged:
        status = "diverged"
    elif not crit["valid"]:
        status = "invalid"
    else:
        status = "converged" if crit["converged"] else "not_converged"
    return dict(status=status, admissible=status in ("converged", "not_converged"), reason=crit["reason"],
                iterations_run=rep.iterations,
                iterations_to_criterion=crit["iteration"] if status == "converged" else None,
                criterion_iteration_id=crit["iteration_id"] if status == "converged" else None,
                iterations_to_residuals=res_only["iteration"] if res_only["valid"] and res_only["converged"] else None,
                n_aligned=crit["n_aligned"], exec_time_s=rep.exec_time_s, diverged=rep.diverged,
                returncode=returncode, missing_monitors=[k for k in names if k not in q],
                final={k: float(v[1][-1]) for k, v in q.items() if len(v[1])})


def enable_potential_foam(system_dir: str) -> None:
    """Add what potentialFoam -writep needs when absent: the Phi solver and potentialFlow controls (fvSolution) and
    the div(div(phi,U)) scheme used to reconstruct pressure (fvSchemes). The SIMPLE solve that follows reads
    neither, so the arm's numerics are otherwise identical to the other arms."""
    fvs = os.path.join(system_dir, "fvSolution")
    txt = open(fvs).read()
    if not re.search(r"^\s*Phi\s*$|^\s*Phi\s*\{", txt, flags=re.M):
        m = re.search(r"^solvers\s*\{", txt, flags=re.M)
        phi = ("\n    Phi\n    {\n        solver          GAMG;\n        smoother        GaussSeidel;\n"
               "        tolerance       1e-06;\n        relTol          0.01;\n    }\n")
        txt = txt[:m.end()] + phi + txt[m.end():]
    if not re.search(r"^potentialFlow\s*\{", txt, flags=re.M):
        txt = txt.rstrip() + "\n\npotentialFlow\n{\n    nNonOrthogonalCorrectors 0;\n}\n"
    open(fvs, "w").write(txt)
    fsc = os.path.join(system_dir, "fvSchemes")
    txt = open(fsc).read()
    if "div(div(phi,U))" not in txt:
        m = re.search(r"^divSchemes\s*\{", txt, flags=re.M)
        txt = txt[:m.end()] + "\n    div(div(phi,U)) Gauss linear;" + txt[m.end():]
        open(fsc, "w").write(txt)
