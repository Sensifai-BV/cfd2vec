"""Encoder-usefulness diagnostics and field-free re-evaluation of one checkpoint on held-out urban cases.

    python scripts/encoder_diagnostics.py --ckpt runs/S_urban_v2_masked/best.pt \
        --shards /path/cfd2vec_shards_v2 --out results/diagnostics_S_urban_v2_masked

Outputs (CSV per diagnostic + summary.json):
    label_leak.csv        geometry-only rel-L2 with stored labels visible vs removed, on the pretraining eval cases
    memory.csv            own / zeroed / swapped spatial memory
    prior.csv             geometry only / prior alone / model with prior
    gradients.json        gradient norm per module group on one training sample
    readout.csv           nested leave-one-out ridge R^2 (with 95 % case-bootstrap interval) of case-level targets
                          from the trained embedding, its CLS and spatial-mean halves, a random encoder and geometry
                          features; the selection-biased single-level value is kept for comparison
    readout_features.npz  the feature matrices used by readout.csv
    invariance.csv        change of the prediction under a new context seed, reordered input points and a 90 degree
                          rotation about z (velocity rotated back), relative to the seed-0 prediction
    provenance.json       command, checkpoint / manifest / source hashes, versions and wall time
Every file in --out is written by one run of this script (the directory is cleared of these names first).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from cfd2vec.api import CFD2vec
from cfd2vec.eval import diagnostics as D
from cfd2vec.schema import Case
from cfd2vec.tasks.predict import spec_from_config
from cfd2vec.data.sampling import rotate_case_z, subsample
from cfd2vec.eval.metrics import rel_l2
from cfd2vec.tasks.predict import predict_case
from cfd2vec.train.dataset import file_sha256, make_sample, shard_paths, to_torch
from cfd2vec.train.pretrain import source_fingerprint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--shards", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--eval-cases", type=int, default=12)
ap.add_argument("--readout-split", default="val"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--invariance-cases", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
OUTPUTS = ("label_leak.csv", "memory.csv", "prior.csv", "gradients.json", "readout.csv", "readout_split.csv",
           "readout_features.npz", "invariance.csv", "summary.json", "provenance.json")
for f in OUTPUTS:                                   # no stale file can survive from an earlier run
    if os.path.exists(os.path.join(a.out, f)):
        os.remove(os.path.join(a.out, f))
t0 = time.time()
man = os.path.join(a.shards, "manifest_aero_urban.parquet")
prov = dict(command=" ".join([os.path.basename(sys.executable)] + sys.argv), started=time.strftime("%Y-%m-%d %H:%M:%S"),
            ckpt_sha256=file_sha256(a.ckpt), manifest=man, manifest_sha256=file_sha256(man), source=source_fingerprint(),
            torch=torch.__version__, python=platform.python_version(), host=platform.platform(), threads=a.threads)
m = CFD2vec.from_pretrained(a.ckpt, device="cpu")
net = m.net
val = shard_paths(man, "val")
eval_p = sorted(np.random.default_rng(a.seed).choice(val, min(a.eval_cases, len(val)), replace=False).tolist())
cases = [Case.load(p) for p in eval_p]
summary = dict(ckpt=a.ckpt, input_schema=net.schema, step=m.meta.get("step"), eval_cases=[c.case_id for c in cases])


def med(df):
    return {k: float(df[k].median()) for k in df.columns if k != "case_id"}


def say(msg):
    print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)


df = pd.DataFrame(D.label_leak_check(net, cases, seed=a.seed)); df.to_csv(os.path.join(a.out, "label_leak.csv"), index=False)
summary["label_leak_median"] = med(df); say("label leak done")
df = pd.DataFrame(D.memory_dependence(net, cases, seed=a.seed)); df.to_csv(os.path.join(a.out, "memory.csv"), index=False)
summary["memory_median"] = med(df); say("memory dependence done")
df = pd.DataFrame(D.prior_dependence(net, cases, seed=a.seed)); df.to_csv(os.path.join(a.out, "prior.csv"), index=False)
summary["prior_median"] = med(df); say("prior dependence done")

cfg = net.cfg
cfg.grad_checkpoint, cfg.decode_chunk = True, 2048
spec = spec_from_config(cfg)
tr = shard_paths(man, "train")[0]
b = {k: v[None] for k, v in to_torch(make_sample(Case.load(tr), spec, np.random.default_rng(a.seed))).items()}
b.pop("sample_key", None)
summary["gradients"] = D.gradient_transport(net, b); say("gradients done")
json.dump(summary["gradients"], open(os.path.join(a.out, "gradients.json"), "w"), indent=1)

rp = shard_paths(man, a.readout_split)
rc = [Case.load(p) for p in rp]
E_tr = D.embeddings(net, rc, seed=a.seed)
d = net.cfg.d_model
E = {"trained_embedding": E_tr, "trained_cls_only": E_tr[:, :d], "trained_spatial_mean_only": E_tr[:, d:]}
say("trained embeddings done")
E["random_encoder_embedding"] = D.embeddings(D.random_like(net, seed=a.seed), rc, seed=a.seed)
say("random embeddings done")
E["geometry_features"] = np.stack([D.geometry_features(c) for c in rc])
np.savez(os.path.join(a.out, "readout_features.npz"), case_ids=np.array([c.case_id for c in rc]), **E)
df = pd.DataFrame(D.frozen_readout(E, rc)); df.to_csv(os.path.join(a.out, "readout.csv"), index=False)
summary["readout"] = df.to_dict("records"); summary["readout_split"] = a.readout_split; summary["n_readout"] = len(rc)
say("readout done")

rows = []
th = np.pi / 2
Rb = np.array([[np.cos(-th), -np.sin(-th), 0.0], [np.sin(-th), np.cos(-th), 0.0], [0.0, 0.0, 1.0]])
for c in cases[:a.invariance_cases]:
    rng = np.random.default_rng(a.seed)
    q = subsample(c, np.sort(rng.choice(c.n, min(4096, c.n), replace=False)))
    base = predict_case(net, c, query=q, seed=a.seed)["fields"]
    variants = dict(context_seed_1=predict_case(net, c, query=q, seed=a.seed + 1)["fields"],
                    reorder=predict_case(net, subsample(c, np.random.default_rng(1).permutation(c.n)), query=q,
                                         seed=a.seed)["fields"])
    rot = predict_case(net, rotate_case_z(c, th), query=rotate_case_z(q, th), seed=a.seed)["fields"].copy()
    rot[:, 0:3] = rot[:, 0:3] @ Rb.T
    variants["rotate_z_90"] = rot
    r = dict(case_id=c.case_id)
    for name, x in variants.items():
        r.update({f"{name}_{k}": v for k, v in rel_l2(x, base, c.presence).items()})
    r.update({f"truth_{k}": v for k, v in rel_l2(base, q.fields, c.presence).items()})
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "invariance.csv"), index=False)
summary["invariance_median"] = med(df); say("invariance done")

summary["wall_s"] = round(time.time() - t0, 1)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
prov.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=summary["wall_s"],
            outputs={f: file_sha256(os.path.join(a.out, f)) for f in OUTPUTS
                     if f != "provenance.json" and os.path.exists(os.path.join(a.out, f))})
json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
say("done")
