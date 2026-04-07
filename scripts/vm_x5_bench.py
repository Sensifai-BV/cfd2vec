"""Run the prepared warm-start benchmark arms serially next to the solver (python3 + numpy + OpenFOAM only).

    python3 vm_x5_bench.py <bench_root>/<case> [--arms cold,mapFields_U,...] [--max-iterations N]

Per arm: disable the residualControl stop, run the fixed iteration budget with drag and pedestrian-probe monitors,
then score iterations to the protocol criterion. Writes <arm>/x5.json and <case>/x5_results.json.
Each arm carries `status` and `admissible`; only admissible arms (converged / not_converged on complete, aligned
evidence) belong in a success summary. Invalid, diverged and failed arms are reported, never scored.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
from cfd2vec.solvers.openfoam import OpenFOAMAdapter  # noqa: E402
from cfd2vec.tasks.benchmark import disable_residual_stop, install_monitors, monitor_functions, score_arm  # noqa: E402
from cfd2vec.tasks.warmstart import time_dirs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("case_root"); ap.add_argument("--arms", default="")
ap.add_argument("--max-iterations", type=int, default=0)
ap.add_argument("--env", default="source /opt/openfoam14/etc/bashrc")
a = ap.parse_args()
P = json.load(open(os.path.join(a.case_root, "protocol.json")))
probes = json.load(open(os.path.join(a.case_root, "probes.json")))
n_max = a.max_iterations or int(P["max_iterations"])
arms = [x for x in (a.arms.split(",") if a.arms else [d["name"] for d in P["arms"]])]
ad = OpenFOAMAdapter()
results = []
for arm in arms:                                   # serial: one solver at a time
    d = os.path.join(a.case_root, arm)
    spec = json.load(open(os.path.join(d, "arm.json")))
    stale = time_dirs(d) + (["postProcessing"] if os.path.isdir(os.path.join(d, "postProcessing")) else [])
    if stale:                                      # earlier output would mix into this arm's logs and monitors
        results.append(dict(arm=arm, status="invalid", admissible=False,
                            error=f"arm directory holds earlier output {stale[:5]}; prepare a fresh arm"))
        continue
    disable_residual_stop(os.path.join(d, "system", "fvSolution"))
    install_monitors(os.path.join(d, "system", "controlDict"), monitor_functions(probes["walls"], np.asarray(probes["points"])))
    ad.set_control(d, {"endTime": n_max, "writeControl": "timeStep", "writeInterval": n_max}, a.env)
    t0 = time.time(); pre_s = 0.0
    if spec.get("pre"):
        for stale in ("Phi", "phi"):                 # outputs of an earlier pre-processing attempt
            if os.path.exists(os.path.join(d, "0", stale)):
                os.remove(os.path.join(d, "0", stale))
        r = ad._sh(d, spec["pre"] + " > log.pre 2>&1", a.env); pre_s = time.time() - t0
        if r.returncode != 0:
            results.append(dict(arm=arm, status="failed", admissible=False,
                                error=f"pre-processing failed: {spec['pre']}")); continue
    rep = ad.run(d, ("foamRun",), log_name="log.run", env_setup=a.env)
    wall = time.time() - t0
    s = score_arm(d, os.path.join(d, "log.run"), P, returncode=rep.returncode)
    s.update(arm=arm, pre_s=round(pre_s, 2), charged_s=spec.get("charge_s", 0.0), wall_s=round(wall, 2),
             max_iterations=n_max)
    json.dump(s, open(os.path.join(d, "x5.json"), "w"), indent=1)
    results.append(s)
    print(json.dumps(s), flush=True)
json.dump(results, open(os.path.join(a.case_root, "x5_results.json"), "w"), indent=1)
bad = [r.get("arm") for r in results if not r.get("admissible")]
if bad:
    print(f"not admissible for the X5 summary: {bad}", flush=True)
