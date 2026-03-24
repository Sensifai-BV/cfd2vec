"""Run warm-start arms next to the solver (needs only python3 + numpy and an OpenFOAM installation).

    python3 vm_warmstart_bench.py <arm_dir> [--cold] [--env "source /opt/openfoam14/etc/bashrc"]

Every arm uses the unchanged solver, mesh, schemes and residualControl. Warm arms run through the residual-checked
safety controller (probe 25 iterations, continue or fall back to cold); the probe is charged to the arm.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cfd2vec.solvers.openfoam import OpenFOAMAdapter  # noqa: E402
from cfd2vec.tasks.warmstart import SafetyPolicy, run_with_safety  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arm_dir"); ap.add_argument("--cold", action="store_true")
ap.add_argument("--env", default="source /opt/openfoam14/etc/bashrc")
ap.add_argument("--probe", type=int, default=25)
a = ap.parse_args()
t0 = time.time()
rep = run_with_safety(OpenFOAMAdapter(), a.arm_dir, SafetyPolicy(enabled=not a.cold, probe_iter=a.probe), env_setup=a.env)
rep.update(arm=os.path.basename(os.path.normpath(a.arm_dir)), wall_s=round(time.time() - t0, 2), safety=not a.cold)
json.dump(rep, open(os.path.join(a.arm_dir, "bench.json"), "w"), indent=1)
print(json.dumps(rep))
