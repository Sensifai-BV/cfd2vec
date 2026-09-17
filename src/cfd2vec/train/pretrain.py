"""Masked field modelling pretraining (and the supervised-regression control with the same network).

Devices: cuda, mps and cpu (`--device auto` picks the first available). Precision: fp32 everywhere by default on
mps and cpu; bf16 autocast on cuda. `--precision` overrides (fp16 uses a gradient scaler).

Training pool: the explicit `manifests` list of the config, or one <shards>/manifest_<source>.parquet per entry of
`sources`; combined and validated by `dataset.load_pool` (fails on a missing manifest or source, duplicate cases,
split or family overlap, or a held-out source listed in `heldout_sources`). An epoch is one seeded permutation of
the whole training pool (or, with `source_weights`, a seeded weighted draw), so every case is eligible whatever the
batch size.

Run directory (everything needed to monitor or resume a run):
    run.json       resolved configuration, device, parameter count, pool sizes, epoch length and provenance
                   (source hash, git revision if any, config / manifest / channel-stats hashes, pool hash, schema)
    pool.csv       the resolved pool (case id, source, split, tier, shard)
    coverage.json  distinct cases drawn and draws, overall and by source / tier
    status.json    live state, rewritten atomically every log interval: step, ETA, throughput, loss, memory, last eval
    log.jsonl      one record per log interval and per evaluation
    train.log      human-readable progress (also printed)
    last.pt        model, optimiser, step, sampler position, RNG and scaler state, coverage (resume with --resume)
    best.pt        weights with the lowest held-out reconstruction loss
Ctrl-C or SIGTERM finishes the current step, saves last.pt without evaluating and exits cleanly; a second Ctrl-C
aborts.
"""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import time
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..eval.metrics import rel_l2, summarise
from ..model.network import CFD2vecNet, ModelConfig, count_parameters
from ..paths import expand, expand_config_paths
from ..schema import Case
from ..tasks.predict import predict_case, spec_from_config
from .dataset import ShardDataset, numpy_collate, to_torch, tokeniser_stats


# ----------------------------------------------------------------------------------------------- device helpers
def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        if pref == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but torch.backends.mps.is_available() is False "
                               f"(built={torch.backends.mps.is_built()})")
        if pref.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def limit_memory(dev: str, fraction: float) -> None:
    """Cap accelerator memory so an oversized batch raises an out-of-memory error instead of swapping.
    On mps the fraction is relative to the recommended working set reported by the driver."""
    if not fraction or fraction <= 0:
        return
    if dev == "mps":
        torch.mps.set_per_process_memory_fraction(float(fraction))
    elif dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(min(float(fraction), 1.0))


def default_precision(dev: str) -> str:
    return "bf16" if dev.startswith("cuda") else "fp32"


def autocast_ctx(dev: str, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=dev.split(":")[0], dtype=dtype)


def memory_gb(dev: str) -> dict:
    try:
        if dev.startswith("cuda"):
            return dict(allocated=torch.cuda.memory_allocated() / 1e9, peak=torch.cuda.max_memory_allocated() / 1e9)
        if dev == "mps":
            return dict(allocated=torch.mps.current_allocated_memory() / 1e9,
                        driver=torch.mps.driver_allocated_memory() / 1e9)
    except Exception:  # noqa: BLE001
        pass
    return {}


def sync(dev: str):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def empty_cache(dev: str):
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * 0.5 * (1 + math.cos(math.pi * t))


def should_evaluate(step: int, eval_every: int, last: bool, stopped: bool) -> bool:
    """Evaluate at an evaluation boundary or at the final step, never after a stop signal: an interrupted run saves
    and exits without the evaluation pass, whatever step it stopped on."""
    return (not stopped) and (step % eval_every == 0 or last)


def _worker_init(_):
    torch.set_num_threads(1)          # token geometry runs in workers; one thread each avoids oversubscription


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate_zero_shot(model, paths, device, n_points=8192, use_prior=False, seed=0):
    """Fully masked prediction error on held-out cases, on a fixed random subset of each case's points."""
    from ..data.sampling import subsample
    rows = []
    rng = np.random.default_rng(seed)
    for p in paths:
        c = Case.load(p)
        q = subsample(c, np.sort(rng.choice(c.n, min(n_points, c.n), replace=False)))
        out = predict_case(model, c, query=q, use_prior=use_prior, seed=seed, device=device)
        rows.append(rel_l2(out["fields"], q.fields, c.presence))
    return summarise(rows)


@torch.no_grad()
def held_out_recon(model, ds_val, dev, precision):
    vl = []
    for i in range(len(ds_val)):
        vb = {k: v[None].to(dev) for k, v in ds_val[i].items()}
        with autocast_ctx(dev, precision):
            pred = model(vb)[0]
        vl.append(float(model.loss(pred.float(), vb)[0]))
    return float(np.mean(vl))


# ----------------------------------------------------------------------------------------------- provenance
def source_fingerprint() -> dict:
    """sha256 over the package's Python sources (sorted relative paths and contents), plus the git revision and
    working-tree state when the package sits in a git checkout with commits."""
    import hashlib
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256(); n = 0
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith((".py", ".yaml")):
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).encode()); h.update(open(p, "rb").read()); n += 1
    out = dict(source_sha256=h.hexdigest(), n_files=n, git_revision=None, git_dirty=None)
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if rev.returncode == 0:
            out["git_revision"] = rev.stdout.strip()
            st = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
            out["git_dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def resolve_manifests(y: dict) -> list:
    """Explicit `manifests` list, or one <shards>/manifest_<source>.parquet per entry of `sources`."""
    if y.get("manifests"):
        return [expand(m) for m in y["manifests"]]
    return [os.path.join(expand(y["shards"]), f"manifest_{s}.parquet") for s in y.get("sources", ["aero_urban"])]


def device_rng_state(dev: str):
    try:
        if dev.startswith("cuda"):
            return torch.cuda.get_rng_state_all()
        if dev == "mps":
            return torch.mps.get_rng_state()
    except Exception:  # noqa: BLE001
        pass
    return None


def set_device_rng_state(dev: str, state) -> bool:
    if state is None:
        return False
    try:
        if dev.startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in state]); return True
        if dev == "mps":
            torch.mps.set_rng_state(state.cpu()); return True
    except Exception:  # noqa: BLE001
        pass
    return False


def coverage_summary(seen: np.ndarray, pool) -> dict:
    """Realised case coverage: distinct cases drawn and draws, overall and by source / tier."""
    out = dict(n_pool=int(len(seen)), n_distinct=int((seen > 0).sum()), draws=int(seen.sum()),
               fraction=float((seen > 0).mean()) if len(seen) else 0.0)
    for col in ("source", "tier"):
        if col in pool.columns:
            g = {}
            for v, idx in pool.groupby(pool[col].astype(str)).indices.items():
                g[v] = dict(n=int(len(idx)), distinct=int((seen[idx] > 0).sum()), draws=int(seen[idx].sum()))
            out[f"by_{col}"] = g
    return out


# ----------------------------------------------------------------------------------------------- main loop
def pretrain(config: str, out_dir: str, objective: str = "masked", device: str = "auto", threads: int = 0,
             max_minutes: float | None = None, max_steps: int | None = None, init: str | None = None,
             resume: bool = False, precision: str | None = None, num_workers: int | None = None,
             batch_size: int | None = None, grad_checkpoint: bool | None = None):
    """Resume semantics: last.pt stores model, optimiser, step, epoch, the position inside the epoch's key stream,
    CPU / NumPy / accelerator RNG, gradient-scaler state and case coverage. Sample keys fix every sample, so a resumed
    run continues the same sample sequence; it is bit-identical only where the accelerator kernels are deterministic."""
    from .dataset import KeyedSampler, SAMPLER_VERSION, file_sha256, load_pool, pool_fingerprint
    from ..model.network import migrate_state_dict
    y = expand_config_paths(yaml.safe_load(open(config)))
    cfg = ModelConfig.from_yaml(y["model"])
    if grad_checkpoint is not None:           # memory / speed trade-off only; weights and schedule are unaffected
        cfg.grad_checkpoint = grad_checkpoint
    stats = json.load(open(y["channel_stats"]))
    seed = int(y.get("seed", 0))
    manifests = resolve_manifests(y)
    pool = load_pool(manifests, sources=y.get("sources"), exclude_sources=y.get("heldout_sources", []))
    train_pool = pool[pool["split"] == "train"].reset_index(drop=True)
    train_p, val_p = train_pool["shard"].tolist(), sorted(pool.loc[pool["split"] == "val", "shard"].tolist())
    if not train_p or not val_p:
        raise ValueError(f"pool has {len(train_p)} training and {len(val_p)} validation cases")
    rng = np.random.default_rng(seed)
    eval_p = sorted(rng.choice(val_p, min(y.get("eval_cases", 12), len(val_p)), replace=False).tolist())
    dev = pick_device(device)
    prec = precision or y.get("precision") or default_precision(dev)
    limit_memory(dev, float(y.get("memory_fraction", 0.9)))
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    mk = y.get("mask", {})
    spec = spec_from_config(cfg, mask_ratios=mk.get("ratios", (0.6, 0.9, 1.0)), mask_probs=mk.get("probs", (0.5, 0.25, 0.25)),
                            mask_modes=mk.get("modes", ("tokens", "near_wall", "wake", "boxes")),
                            mode_probs=mk.get("mode_probs", (0.4, 0.2, 0.2, 0.2)),
                            fixed_ratio=1.0 if objective == "supervised" else None,
                            prior_prob=y.get("prior_prob", 0.5))
    # YAML 1.1 reads exponents without a sign (1.0e9) as strings: cast every numeric setting explicitly
    bs = int(batch_size or y.get("batch_size", 8))
    steps = int(max_steps or y.get("max_steps", 10000))
    minutes = float(max_minutes or y.get("max_minutes", 1e9))
    log_every = int(y.get("log_every", 25))
    y["eval_every"] = int(y.get("eval_every", 500))
    o_raw = y.get("optim", {})
    y["optim"] = {k: float(v) for k, v in o_raw.items()}
    weights = y.get("source_weights")
    ds = ShardDataset(train_p, spec, seed=seed)
    ds.numpy_out = True
    groups = train_pool["source"].astype(str).tolist() if weights else None
    sampler = KeyedSampler(len(train_p), seed=seed, groups=groups,
                           weights={str(k): float(v) for k, v in weights.items()} if weights else None)
    epoch_len = sampler.epoch_length()
    if epoch_len < bs:
        raise ValueError(f"epoch length {epoch_len} is smaller than the batch size {bs}")
    n_vr = y.get("val_recon_cases", 0) or len(val_p)          # held-out reconstruction set (0 = all)
    vr_paths = sorted(np.random.default_rng(1).choice(val_p, min(n_vr, len(val_p)), replace=False).tolist())
    ds_val = ShardDataset(vr_paths, spec, seed=12345)          # fixed keys (i, 0, 0): identical inputs every eval
    nw = num_workers if num_workers is not None else y.get("num_workers", 4)
    # the loader draws a worker base seed from its generator every time an epoch starts; a private generator keeps
    # those draws out of the global torch RNG, which the model uses (closure dropout), so resuming mid-epoch does
    # not shift the model's random stream. Sample randomness itself comes from the sample keys.
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=nw, persistent_workers=nw > 0, drop_last=True,
                    collate_fn=numpy_collate, worker_init_fn=_worker_init, prefetch_factor=4 if nw > 0 else None,
                    generator=torch.Generator().manual_seed(seed))
    model = CFD2vecNet(cfg, stats["mean"], stats["std"]).to(dev)
    o = y.get("optim", {})
    opt = torch.optim.AdamW(model.parameters(), lr=o.get("lr", 3e-4), weight_decay=o.get("weight_decay", 0.05),
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(dev.split(":")[0]) if prec == "fp16" else None
    fingerprint = pool_fingerprint(pool)
    prov = dict(source=source_fingerprint(), config_path=os.path.abspath(config), config_sha256=file_sha256(config),
                model_config_sha256=file_sha256(y["model"]), channel_stats_sha256=file_sha256(y["channel_stats"]),
                manifests={m: file_sha256(m) for m in manifests}, pool_sha256=fingerprint,
                sampler=SAMPLER_VERSION, input_schema=cfg.input_schema)
    step, epoch, cursor, best, elapsed0 = 0, 0, 0, float("inf"), 0.0
    seen = np.zeros(len(train_p), np.int64)
    last_path = os.path.join(out_dir, "last.pt")
    resume_note = None
    if resume and os.path.exists(last_path):
        # load on CPU: model and optimiser copy their tensors to the device; the RNG state must stay a CPU tensor
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        ck_schema = ModelConfig.from_checkpoint(ck.get("cfg", {})).input_schema
        if ck_schema != cfg.input_schema:
            raise ValueError(f"{last_path} uses input schema {ck_schema}, the config builds schema "
                             f"{cfg.input_schema}; start a new run (optionally --init from it)")
        old_fp = (ck.get("provenance") or {}).get("pool_sha256")
        if old_fp is not None and old_fp != fingerprint:
            raise ValueError("the training pool changed since this run started (pool_sha256 differs); "
                             "start a new run instead of resuming")
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        step, epoch, best = ck.get("step", 0), ck.get("epoch", 0), ck.get("best", float("inf"))
        elapsed0 = ck.get("elapsed_min", 0.0)
        smp = ck.get("sampler")
        if smp and smp.get("version") == SAMPLER_VERSION and smp.get("epoch_length") == epoch_len:
            epoch, cursor = int(smp["epoch"]), int(smp["cursor"])
            if smp.get("seen") is not None and len(smp["seen"]) == len(seen):
                seen = np.asarray(smp["seen"], np.int64)
        else:
            resume_note = "checkpoint has no compatible sampler state: the key stream restarts at the next epoch"
            epoch, cursor = epoch + 1 if smp is None and step else epoch, 0
        if "rng" in ck:
            torch.set_rng_state(ck["rng"]["torch"].cpu().to(torch.uint8)); np.random.set_state(ck["rng"]["numpy"])
            set_device_rng_state(dev, ck["rng"].get("device"))
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
    elif init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        src_schema = ModelConfig.from_checkpoint(ck.get("cfg", {})).input_schema
        model.load_state_dict(migrate_state_dict(ck["model"], src_schema, cfg.input_schema))
        prov["init"] = dict(path=os.path.abspath(init), sha256=file_sha256(init), input_schema=src_schema)
    # realised token support on real shards (coverage, effective radii per scale) next to the nominal radii
    tok_stats = tokeniser_stats(train_p, spec, n_cases=int(y.get("tokeniser_stats_cases", 8)), seed=seed)
    meta = dict(config=config, objective=objective, device=dev, precision=prec, params=count_parameters(model),
                model=cfg.to_dict(), tokeniser=tok_stats, resolved_config=y, n_train=len(train_p), n_val=len(val_p), eval_cases=eval_p,
                batch_size=bs, epoch_length=epoch_len, steps_per_epoch=epoch_len // bs,
                keys_dropped_per_epoch=epoch_len % bs,     # drop_last: these keys of each epoch are not consumed
                nominal_passes=round(steps * bs / max(epoch_len, 1), 2), source_weights=weights,
                max_steps=steps, max_minutes=minutes, num_workers=nw, torch=torch.__version__,
                host=platform.platform(), python=platform.python_version(),
                resumed_from_step=step if resume else None, resume_note=resume_note, provenance=prov,
                exact_resume="sample keys, sampler position, CPU/NumPy/accelerator RNG and scaler state are saved; "
                             "bit-identical continuation additionally needs deterministic accelerator kernels")
    _atomic_json(meta, os.path.join(out_dir, "run.json"))
    cols = [c for c in ("case_id", "source", "split", "tier", "shard") if c in pool.columns]
    pool[cols].to_csv(os.path.join(out_dir, "pool.csv"), index=False)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    txt = open(os.path.join(out_dir, "train.log"), "a")

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); txt.write(line + "\n"); txt.flush()

    status = dict(state="starting", objective=objective, device=dev, precision=prec, step=step, max_steps=steps,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid(), last_eval=None, best_val_recon=best)
    status_path = os.path.join(out_dir, "status.json")
    cov_path = os.path.join(out_dir, "coverage.json")
    _atomic_json(status, status_path)

    stop = {"n": 0, "why": None}

    def on_signal(sig, _frm):
        stop["n"] += 1; stop["why"] = signal.Signals(sig).name
        if stop["n"] > 1:
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def save_last(state_note):
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), cfg=cfg.to_dict(), stats=stats,
                        step=step, epoch=epoch, best=best, objective=objective, elapsed_min=elapsed(),
                        sampler=dict(version=SAMPLER_VERSION, epoch=epoch, cursor=cursor, epoch_length=epoch_len,
                                     seen=seen.copy()),
                        scaler=scaler.state_dict() if scaler is not None else None,
                        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                                 device=device_rng_state(dev)),
                        provenance=prov, note=state_note),
                   last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    t0 = time.time()
    elapsed = lambda: elapsed0 + (time.time() - t0) / 60  # noqa: E731
    say(f"{objective} {cfg.name} schema={cfg.input_schema} params={meta['params']:,} device={dev} precision={prec} "
        f"batch={bs} workers={nw} train={len(train_p)} val={len(val_p)} epoch_length={epoch_len} "
        f"start_step={step} epoch={epoch} cursor={cursor}")
    tm = tok_stats["median"]
    say(f"tokeniser on {tok_stats['n_cases']} training shards: coverage {tm['coverage']:.3f} "
        f"(near-wall {tm['coverage_near_wall']:.3f}) | r_eff median scale 0 {tm['r_eff_scale0_q50']:.3f} "
        f"(nominal r1 {spec.r1}) scale 1 {tm['r_eff_scale1_q50']:.3f} (nominal r2 {spec.r2})")
    if resume_note:
        say(resume_note)
    run, n_bad, t_int, s_int, mask_real = [], 0, time.time(), step, []
    status["state"] = "running"
    timed_out = False
    try:
        while step < steps and not stop["n"] and not timed_out:
            sampler.set_epoch(epoch, start=cursor)
            last = False
            for b in dl:
                model.train()
                keys = b.pop("sample_key")
                mask_real.extend(np.asarray(b.pop("mask_realised")).tolist())
                cursor += len(keys)
                np.add.at(seen, keys[:, 0], 1)
                b = {k: v.to(dev, non_blocking=True) for k, v in to_torch(b).items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, o.get("lr", 3e-4), o.get("warmup_steps", 500), steps)
                with autocast_ctx(dev, prec):
                    pred, _ = model(b)
                loss, per = model.loss(pred.float(), b)
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    n_bad += 1
                    say(f"non-finite loss at step {step} (#{n_bad}); step skipped")
                    if n_bad >= 20:
                        status["state"] = "failed: non-finite loss"; raise RuntimeError("20 non-finite losses")
                    continue
                n_bad = 0
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("grad_clip_norm", 1.0))
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("grad_clip_norm", 1.0))
                    opt.step()
                step += 1
                run.append(float(loss.detach()))
                if step % log_every == 0:
                    sync(dev)
                    dt = time.time() - t_int; sps = (step - s_int) / max(dt, 1e-9)
                    t_int, s_int = time.time(), step
                    cov = coverage_summary(seen, train_pool)
                    rec = dict(step=step, epoch=epoch, t_min=elapsed(), loss=float(np.mean(run)), gnorm=float(gn),
                               lr=opt.param_groups[0]["lr"], steps_per_s=sps, samples_per_s=sps * bs,
                               per_channel=[round(float(x), 4) for x in per.cpu()], mem_gb=memory_gb(dev),
                               distinct_cases=cov["n_distinct"],
                               mask_realised=float(np.mean(mask_real)) if mask_real else None)
                    log.write(json.dumps(rec) + "\n"); log.flush(); run = []; mask_real = []
                    _atomic_json(dict(cov, step=step, epoch=epoch), cov_path)
                    eta = (steps - step) / max(sps, 1e-9) / 60
                    status.update(step=step, epoch=epoch, elapsed_min=round(rec["t_min"], 2), eta_min=round(eta, 1),
                                  loss=rec["loss"], lr=rec["lr"], steps_per_s=round(sps, 3),
                                  samples_per_s=round(sps * bs, 2), mem_gb=rec["mem_gb"],
                                  distinct_cases=cov["n_distinct"], updated=time.strftime("%Y-%m-%d %H:%M:%S"))
                    _atomic_json(status, status_path)
                    say(f"step {step}/{steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} {sps * bs:.1f} samples/s "
                        f"cases seen {cov['n_distinct']}/{cov['n_pool']} eta {eta:.0f} min")
                timed_out = elapsed() > minutes
                last = step >= steps or timed_out or bool(stop["n"])
                if should_evaluate(step, int(y["eval_every"]), last, bool(stop["n"])):
                    model.eval()
                    status["state"] = "evaluating"; _atomic_json(status, status_path)
                    vr = held_out_recon(model, ds_val, dev, prec)
                    zs = evaluate_zero_shot(model, eval_p, dev, use_prior=False)
                    sf = evaluate_zero_shot(model, eval_p, dev, use_prior=True)
                    rec = dict(step=step, t_min=elapsed(), val_recon_loss=vr, zero_shot=zs, with_prior=sf)
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    if vr < best:
                        best = vr
                        torch.save(dict(model=model.state_dict(), cfg=cfg.to_dict(), stats=stats, step=step,
                                        objective=objective, provenance=prov), os.path.join(out_dir, "best.pt"))
                    save_last("periodic")
                    empty_cache(dev)
                    status.update(state="running", last_eval=rec, best_val_recon=best)
                    _atomic_json(status, status_path)
                    say(f"eval step {step}: val_recon {vr:.4f} | geometry-only U {zs.get('rel_l2_U', float('nan')):.3f} "
                        f"Cp {zs.get('rel_l2_Cp', float('nan')):.3f} | with prior U {sf.get('rel_l2_U', float('nan')):.3f}")
                if last:
                    break
            if not last:                      # the epoch's key stream is exhausted
                epoch += 1; cursor = 0
        _atomic_json(dict(coverage_summary(seen, train_pool), step=step, epoch=epoch), cov_path)
        save_last("final" if step >= steps else f"stopped ({stop['why'] or 'time limit'})")
        status["state"] = "finished" if step >= steps else f"stopped ({stop['why'] or 'time limit'})"
    except KeyboardInterrupt:
        status["state"] = "aborted"
        raise
    except Exception as e:  # noqa: BLE001
        status["state"] = status["state"] if status["state"].startswith("failed") else f"failed: {e!r}"[:300]
        try:
            save_last("crash")
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        status.update(step=step, elapsed_min=round(elapsed(), 2), updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        _atomic_json(status, status_path)
        say(f"{status['state']} at step {step} after {elapsed():.1f} min")
    return out_dir
