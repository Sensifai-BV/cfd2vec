"""The two empirical acceptance tests of the V1 plan that need real cases, on the urban benchmark cases.

Frame check (voxel cache versus OpenFOAM adapter of the same case):
    python scripts/geometry_path_check.py frames --voxel <cache>/<tier>/<case>.npz --case-dir <bench>/<B>/fine \\
        --U-ref <Uref> --out results/audits/frames_<B>.json

Remesh check (fine versus coarse mesh of one benchmark case, one checkpoint, geometry-only prediction on each mesh,
compared at shared physical locations and against the fine truth):
    python scripts/geometry_path_check.py remesh --ckpt runs/<run>/best.pt --fine <bench>/<B>/fine \\
        --coarse <bench>/<B>/coarse --U-ref <Uref> [--alpha <abl_alpha>] --out results/audits/remesh_<B>.json

Both write one JSON report; nothing is modified. The coarse case is read in the fine case's frame (same L_ref and
origin) so the comparison is about the mesh, not about the frame; the frame itself is what the `frames` check
tests.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

from cfd2vec.eval.consistency import compare_frames, compare_on_shared_points
from cfd2vec.paths import expand
from cfd2vec.schema import Conditioning


def frames(a):
    from cfd2vec.solvers.openfoam import OpenFOAMAdapter
    from cfd2vec.sources import aero_urban
    v = aero_urban.read_voxel_npz(expand(a.voxel))
    vox = aero_urban.voxel_to_case(v, with_prior=False)
    cond = Conditioning(closure="k-epsilon", abl_alpha=a.alpha, ground="stationary", rotation_ok=True)
    of = OpenFOAMAdapter().read_case(expand(a.case_dir), U_ref=a.U_ref, L_ref=None, cond=cond)
    rep = dict(check="frames", voxel=a.voxel, case_dir=a.case_dir, voxel_case=vox.case_id, adapter_case=of.case_id,
               normal_orientation=of.meta.get("normal_orientation"),
               result=compare_frames(vox, of, n=a.points, seed=a.seed))
    return rep


def remesh(a):
    import torch
    from cfd2vec.api import CFD2vec
    from cfd2vec.solvers.openfoam import OpenFOAMAdapter
    torch.set_num_threads(a.threads)
    ad = OpenFOAMAdapter()
    cond = Conditioning(closure="k-epsilon", abl_alpha=a.alpha, ground="stationary", rotation_ok=True)
    fine = ad.read_case(expand(a.fine), U_ref=a.U_ref, L_ref=None, cond=copy.deepcopy(cond), with_fields=a.with_fields)
    coarse = ad.read_case(expand(a.coarse), U_ref=a.U_ref, L_ref=fine.L_ref, cond=copy.deepcopy(cond),
                          origin=np.asarray(fine.meta["origin"]), with_fields=False)
    m = CFD2vec.from_pretrained(expand(a.ckpt), device=a.device)
    pf = m.predict(fine, seed=a.seed)["fields"]
    pc = m.predict(coarse, seed=a.seed)["fields"]
    res = compare_on_shared_points(pf, fine, pc, coarse, n=a.points, seed=a.seed,
                                   max_match_over_L=a.max_match if a.max_match > 0 else None)
    return dict(check="remesh", ckpt=a.ckpt, fine=a.fine, coarse=a.coarse, n_fine=fine.n, n_coarse=coarse.n,
                L_ref=fine.L_ref, input_schema=m.net.schema, result=res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames"); f.add_argument("--voxel", required=True); f.add_argument("--case-dir", required=True)
    r = sub.add_parser("remesh"); r.add_argument("--ckpt", required=True); r.add_argument("--fine", required=True)
    r.add_argument("--coarse", required=True); r.add_argument("--with-fields", action="store_true",
                                                             help="the fine case holds a solution: compare against it")
    r.add_argument("--device", default="cpu"); r.add_argument("--threads", type=int, default=4)
    r.add_argument("--max-match", type=float, default=0.0, help="drop pairs farther apart than this (x L_ref)")
    for p in (f, r):
        p.add_argument("--U-ref", type=float, required=True); p.add_argument("--alpha", type=float)
        p.add_argument("--points", type=int, default=20000); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = frames(a) if a.cmd == "frames" else remesh(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)      # the report is written whatever the outcome
    print(json.dumps(rep["result"], indent=1, default=float))
    status = rep["result"].get("status", "ok")
    if status != "ok":
        print(f"{a.cmd}: {status}: {rep['result'].get('reason', '')}; report written to {a.out}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
