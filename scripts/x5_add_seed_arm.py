"""Add a warm-start arm seeded from externally predicted fields to a prepared X5 case (supplementary arm).

    python scripts/x5_add_seed_arm.py --case B6 --name supp_aero_unet_Up \
        --seed-dir ${CFD2VEC_DATA}/aero_bench/B6/warm_ws/warmstart --fields U,p --charge coarse+rpc

The seed directory holds raw little-endian float32 files in OpenFOAM cell order (<field>.f32; U is N x 3), as written
by an external warm-start model. The arm gets the same mesh, schemes, monitors and protocol as the arms made by
scripts/x5_prepare.py, so vm_x5_bench.py runs and scores it unchanged. The charged time is recorded in arm.json:
`coarse` adds the coarse-twin solve from meta.json, `rpc` adds the model request time recorded next to the seed
(client.json, client_rpc_s), and a number adds that many seconds.
"""
import argparse
import json
import os
import shutil

import numpy as np

from cfd2vec.paths import expand
from cfd2vec.solvers.openfoam import OpenFOAMAdapter

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--seed-dir", required=True)
ap.add_argument("--fields", default="U,p")
ap.add_argument("--charge", default="coarse+rpc", help="'+'-joined terms: coarse, rpc, or seconds")
ap.add_argument("--bench", default="${CFD2VEC_DATA}/aero_bench")
ap.add_argument("--out", default="${CFD2VEC_DATA}/cfd2vec_x5")
a = ap.parse_args()
a.bench, a.out, a.seed_dir = expand(a.bench), expand(a.out), expand(a.seed_dir)
src = os.path.join(a.bench, a.case)
root = os.path.join(a.out, a.case)
if not os.path.exists(os.path.join(root, "protocol.json")):
    raise SystemExit(f"{root}: prepare the case with scripts/x5_prepare.py first")
meta = json.load(open(os.path.join(src, "meta.json")))
n = int(meta["cells_fine"])
fields = {}
for f in a.fields.split(","):
    v = np.fromfile(os.path.join(a.seed_dir, f"{f}.f32"), "<f4")
    v = v.reshape(-1, 3) if f == "U" else v
    if v.shape[0] != n:
        raise SystemExit(f"{f}.f32 has {v.shape[0]} rows, the fine mesh has {n} cells")
    if not np.all(np.isfinite(v)):
        raise SystemExit(f"{f}.f32 holds non-finite values")
    fields[f] = v.astype(np.float64)
client = {}
if os.path.exists(os.path.join(a.seed_dir, "client.json")):
    client = json.load(open(os.path.join(a.seed_dir, "client.json")))
charge, parts = 0.0, {}
for term in a.charge.split("+"):
    if term == "coarse":
        parts["coarse_solve_s"] = float(meta.get("exec_coarse_s", 0.0))
    elif term == "rpc":
        parts["model_request_s"] = float(client.get("client_rpc_s", 0.0))
    elif term:
        parts["fixed_s"] = float(term)
charge = sum(parts.values())
d = os.path.join(root, a.name)
if os.path.exists(d):
    raise SystemExit(f"{d} exists; remove it before preparing the arm again")
os.makedirs(d)
for sub in ("constant", "system", "0"):
    shutil.copytree(os.path.join(src, "fine", sub), os.path.join(d, sub))
for f in ("C", "Ccx", "Ccy", "Ccz", "Vc"):
    if os.path.exists(os.path.join(d, "0", f)):
        os.remove(os.path.join(d, "0", f))
OpenFOAMAdapter().write_initial(d, fields)
json.dump(dict(arm=a.name, pre=None, charge_s=charge, charge_parts=parts, seed_dir=a.seed_dir,
               seed_fields=list(fields), seed_client=client), open(os.path.join(d, "arm.json"), "w"), indent=1)
print(f"{a.case}/{a.name}: seeded {','.join(fields)} on {n} cells; charged {charge:.2f} s {parts}")
