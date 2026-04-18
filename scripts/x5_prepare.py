"""Prepare the solver warm-start benchmark arms for one case (runs where the model runs).

    python scripts/x5_prepare.py --case B1 --ckpt runs/S_urban_masked/best.pt --out <bench_root>

Creates <out>/<case>/<arm>/ for every arm in configs/warmstart_protocol.yaml with identical mesh, schemes and
monitors, writes the initial fields of the model arms, the fixed pedestrian probe set (probes.json) and the protocol
(protocol.json, so the solver side needs no YAML parser). Inference time is recorded per arm (charged to it).
"""
import argparse
import copy
import json
import os
import shutil
import time

import numpy as np
import torch
import yaml

from cfd2vec.api import CFD2vec
from cfd2vec.paths import expand
from cfd2vec.schema import Conditioning
from cfd2vec.solvers.openfoam import OpenFOAMAdapter, patch_types
from cfd2vec.tasks.benchmark import enable_potential_foam, pedestrian_probes
from cfd2vec.tasks.warmstart import SeedPolicy, attach_prior, to_physical

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--bench", default="${CFD2VEC_DATA}/aero_bench")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="${CFD2VEC_DATA}/cfd2vec_x5")
ap.add_argument("--protocol", default="configs/warmstart_protocol.yaml")
ap.add_argument("--device", default="auto"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--z-over-H", type=float, default=0.1)
ap.add_argument("--supplementary", action="store_true", help="also prepare arms outside the frozen protocol")
a = ap.parse_args(); a.bench, a.out = expand(a.bench), expand(a.out)
torch.set_num_threads(a.threads)
P = yaml.safe_load(open(a.protocol))
src = os.path.join(a.bench, a.case)
prm = json.load(open(os.path.join(src, "params.json")))["params"]
cond = Conditioning(closure="k-epsilon", abl_alpha=prm["alpha"], turb_intensity=prm.get("I"), ground="stationary",
                    rotation_ok=True)
ad = OpenFOAMAdapter()
fine = ad.read_case(os.path.join(src, "fine"), U_ref=prm["Uref"], L_ref=None, cond=copy.deepcopy(cond))
fine.cond.log10_re = float(np.log10(fine.U_ref * fine.L_ref / prm["nu"]))
coarse = ad.read_case(os.path.join(src, "coarse"), U_ref=prm["Uref"], L_ref=fine.L_ref, cond=copy.deepcopy(cond),
                      with_fields=True, origin=np.asarray(fine.meta["origin"]))
m = CFD2vec.from_pretrained(a.ckpt, device=a.device)
# predictions (timed; the prior arms also depend on the coarse solve, charged from meta.json)
t = time.time(); geo = m.predict(fine)["fields"]; t_geo = time.time() - t
wp = copy.copy(fine); wp.cond = copy.deepcopy(fine.cond); attach_prior(wp, coarse)
t = time.time(); pri = m.predict(wp, use_prior=True)["fields"]; t_pri = time.time() - t
phys_geo, phys_pri, phys_map = (to_physical(f, c, SeedPolicy()) for f, c in ((geo, fine), (pri, wp), (wp.prior, wp)))
# pedestrian probes in metres (fixed for every arm)
L, o = fine.L_ref, np.asarray(fine.meta["origin"])
pts_m = fine.points * L + o
shp = json.load(open(os.path.join(src, "params.json")))["params"].get("shapes", [])
if shp:                                              # footprint box of the building layout
    r = [0.5 * np.hypot(b.get("lx", 0), b.get("ly", 0)) or b.get("r", 0) for b in shp]
    box = (min(b["cx"] - q for b, q in zip(shp, r)), max(b["cx"] + q for b, q in zip(shp, r)),
           min(b["cy"] - q for b, q in zip(shp, r)), max(b["cy"] + q for b, q in zip(shp, r)))
else:
    box = None
probes = pedestrian_probes(pts_m, fine.wall_dist * L, a.z_over_H * L, 0.5 * float(np.median(fine.cell_size)) * L,
                           float(fine.meta.get("ground_z") or 0.0), n=64, seed=0, xy_box=box)
meta = json.load(open(os.path.join(src, "meta.json")))
walls = [p for p, t in patch_types(os.path.join(src, "fine")).items() if t == "wall" and p != "ground"]
arms = {
    "cold": dict(fields=None, pre="", charge_s=0.0),
    "potentialFoam": dict(fields=None, pre="potentialFoam -writep", charge_s=0.0),
    "mapFields_U": dict(fields={"U": phys_map["U"]}, charge_s=float(meta.get("exec_coarse_s", 0.0))),
    "cfd2vec_U": dict(fields={"U": phys_geo["U"]}, charge_s=t_geo),
    "cfd2vec_Up": dict(fields={"U": phys_geo["U"], "p": phys_geo["p"]}, charge_s=t_geo),
    "cfd2vec_Upkeps": dict(fields={k: phys_geo[k] for k in ("U", "p", "k", "epsilon")}, charge_s=t_geo),
}
if a.supplementary:                                  # not in protocol v1; reported separately
    arms["supp_cfd2vec_prior_U"] = dict(fields={"U": phys_pri["U"]},
                                        charge_s=t_pri + float(meta.get("exec_coarse_s", 0.0)))
root = os.path.join(a.out, a.case)
os.makedirs(root, exist_ok=True)
for name, spec in arms.items():
    d = os.path.join(root, name)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for sub in ("constant", "system", "0"):
        shutil.copytree(os.path.join(src, "fine", sub), os.path.join(d, sub))
    for f in ("C", "Ccx", "Ccy", "Ccz", "Vc"):
        if os.path.exists(os.path.join(d, "0", f)):
            os.remove(os.path.join(d, "0", f))
    if spec["fields"]:
        ad.write_initial(d, spec["fields"])
    if spec.get("pre", "").startswith("potentialFoam"):
        enable_potential_foam(os.path.join(d, "system"))
    json.dump(dict(arm=name, pre=spec.get("pre"), charge_s=spec["charge_s"]), open(os.path.join(d, "arm.json"), "w"))
json.dump(dict(points=probes.tolist(), walls=walls, z_m=a.z_over_H * L, xy_box=box), open(os.path.join(root, "probes.json"), "w"))
json.dump(P, open(os.path.join(root, "protocol.json"), "w"), indent=1)
json.dump(dict(case=a.case, ckpt=a.ckpt, L_ref=L, U_ref=fine.U_ref, log10_re=fine.cond.log10_re,
               inference_s=dict(geometry_only=t_geo, with_prior=t_pri), n_cells=fine.n, arms=list(arms)),
          open(os.path.join(root, "prepare.json"), "w"), indent=1)
print(f"prepared {len(arms)} arms in {root}; {len(probes)} probes at z = {a.z_over_H * L:.3f} m; "
      f"inference {t_geo:.1f} s / {t_pri:.1f} s")
