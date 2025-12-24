"""Audit that one training epoch consumes every case of the pretraining pool.

    PYTHONPATH=src python scripts/audit_full_pool_epoch.py --config configs/pretrain_S_urban.yaml \
        --out results/audits/full_pool_epoch

Runs the real data pipeline (manifest, pool validation, keyed sampler, persistent workers, sample builder) with a
small model on CPU for exactly one epoch (epoch_length // batch_size steps), then compares coverage.json with the
pool. Writes the run directory (run.json, pool.csv, coverage.json, log.jsonl, train.log; the model checkpoints are
removed) and audit.json with the verdict, command and hashes. The model and its losses are not a result.
"""
import argparse
import json
import os
import platform
import sys
import time

import yaml

from cfd2vec.train.dataset import file_sha256, load_pool
from cfd2vec.train.pretrain import pretrain, resolve_manifests, source_fingerprint

SMALL_MODEL = dict(name="cfd2vec-audit", d_model=128, n_layers=2, n_heads=4,
                   tokenizer=dict(n_tokens_scale1=256, n_tokens_scale2=128, r1=0.05, r2=0.25, k_neighbors=16,
                                  pointnet_hidden=[32, 64]),
                   decoder=dict(n_cross_layers=1, mlp_ratio=4), data=dict(n_context=8192, n_query=1024))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_S_urban.yaml"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=2); ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run_dir = os.path.join(a.out, "run")
    if os.path.exists(os.path.join(run_dir, "last.pt")):
        raise SystemExit(f"{run_dir} already holds a run; choose a fresh --out")
    y = yaml.safe_load(open(a.config))
    mc = os.path.join(a.out, "model_audit.yaml"); yaml.safe_dump(SMALL_MODEL, open(mc, "w"))
    y.update(model=mc, batch_size=a.batch_size, num_workers=a.num_workers, memory_fraction=0, eval_cases=4,
             val_recon_cases=4)
    y["optim"] = dict(y.get("optim", {}), warmup_steps=20)
    pool = load_pool(resolve_manifests(y), sources=y.get("sources"), exclude_sources=y.get("heldout_sources", []))
    n_train = int((pool["split"] == "train").sum())
    if y.get("source_weights"):
        raise SystemExit("the audit covers proportional sampling; remove source_weights from the config")
    steps = n_train // a.batch_size                        # one epoch of consumed keys
    t0 = time.time()
    cfg = os.path.join(a.out, "pretrain_audit.yaml")
    yaml.safe_dump(dict(y, max_steps=steps, log_every=max(1, steps // 3), eval_every=steps), open(cfg, "w"))
    pretrain(cfg, run_dir, device="cpu", threads=a.threads)
    cov = json.load(open(os.path.join(run_dir, "coverage.json")))
    run = json.load(open(os.path.join(run_dir, "run.json")))
    expected = run["epoch_length"] - run["keys_dropped_per_epoch"]
    ok = cov["n_distinct"] == expected and cov["draws"] == expected and run["n_train"] == cov["n_pool"]
    for f in ("last.pt", "best.pt"):                       # weights of the audit model are not kept
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    audit = dict(passed=bool(ok), n_pool=cov["n_pool"], n_distinct=cov["n_distinct"], draws=cov["draws"],
                 expected_distinct=expected, keys_dropped_per_epoch=run["keys_dropped_per_epoch"],
                 steps=steps, batch_size=a.batch_size, num_workers=a.num_workers, by_source=cov.get("by_source"),
                 by_tier=cov.get("by_tier"), command=" ".join([os.path.basename(sys.executable)] + sys.argv),
                 config=a.config, config_sha256=file_sha256(a.config), pool_sha256=run["provenance"]["pool_sha256"],
                 manifests=run["provenance"]["manifests"], source=source_fingerprint(), host=platform.platform(),
                 wall_s=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(audit, open(os.path.join(a.out, "audit.json"), "w"), indent=1)
    print(json.dumps({k: audit[k] for k in ("passed", "n_pool", "n_distinct", "draws", "by_tier", "wall_s")}))
    return 0 if ok else 1


if __name__ == "__main__":                   # required: data-loader workers re-import this module when spawned
    sys.exit(main())
