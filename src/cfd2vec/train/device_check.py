"""Accelerator check before a long run: availability, numerical parity with CPU, throughput and memory per batch size.

Progress is printed stage by stage. Accelerator memory is capped (`memory_fraction`), so a batch that does not fit
fails fast with an out-of-memory error instead of swapping; the sweep stops at the first failure or when a step
exceeds `max_step_s`. Writes runs/device_check.json.
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import torch

from ..model.network import CFD2vecNet, ModelConfig, count_parameters
from ..schema import Case
from ..tasks.predict import spec_from_config
from .dataset import make_sample, shard_paths, to_torch
from .pretrain import autocast_ctx, empty_cache, limit_memory, memory_gb, pick_device, sync


def _say(msg):
    print(f"[device-check {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device_check(model_config: str, stats_path: str, manifest: str, device: str = "auto",
                 batch_sizes=(1, 2, 4), n_steps: int = 3, out: str = "runs/device_check.json",
                 memory_fraction: float = 0.9, max_step_s: float = 60.0, precision: str = "fp32",
                 grad_checkpoint: bool | None = None) -> dict:
    t_all = time.time()
    dev = pick_device(device)
    cfg = ModelConfig.from_yaml(model_config)
    if grad_checkpoint is not None:
        cfg.grad_checkpoint = grad_checkpoint
    st = json.load(open(stats_path))
    rep = dict(device=dev, torch=torch.__version__, host=platform.platform(), model=cfg.name,
               cuda=torch.cuda.is_available(), mps_built=torch.backends.mps.is_built(),
               mps_available=torch.backends.mps.is_available(), memory_fraction=memory_fraction,
               grad_checkpoint=cfg.grad_checkpoint, decode_chunk=cfg.decode_chunk, precision=precision)
    _say(f"device={dev} torch={torch.__version__} mps_available={rep['mps_available']} cuda={rep['cuda']} "
         f"precision={precision}")
    limit_memory(dev, memory_fraction)
    if dev == "mps":
        rep["mps_recommended_working_set_gb"] = round(torch.mps.recommended_max_memory() / 1e9, 2) \
            if hasattr(torch.mps, "recommended_max_memory") else None
        _say(f"memory cap: {memory_fraction} x recommended working set "
             f"({rep['mps_recommended_working_set_gb']} GB)")
    torch.manual_seed(0)
    net = CFD2vecNet(cfg, st["mean"], st["std"])
    rep["params"] = count_parameters(net)
    _say(f"model {cfg.name}: {rep['params']:,} parameters, tokens {cfg.n_tokens_scale1}+{cfg.n_tokens_scale2}, "
         f"context {cfg.n_context}, queries {cfg.n_query}")
    paths = shard_paths(manifest, "train")[:max(batch_sizes)]
    spec = spec_from_config(cfg)
    rng = np.random.default_rng(0)
    t = time.time()
    samples = [to_torch(make_sample(Case.load(p), spec, rng)) for p in paths]
    rep["sample_prep_s"] = round((time.time() - t) / len(samples), 3)
    _say(f"prepared {len(samples)} samples, {rep['sample_prep_s']} s each (data-loader workers do this in parallel)")
    batch = lambda n: {k: torch.stack([s[k] for s in samples[:n]]) for k in samples[0]}  # noqa: E731

    # parity: same weights and batch on CPU and on the device (head perturbed so the output is non-trivial)
    with torch.no_grad():
        torch.nn.init.normal_(net.head.weight, std=0.02)
        b1 = batch(1)
        t = time.time(); ref = net.eval()(b1)[0]; t_cpu = time.time() - t
        net_d = CFD2vecNet(cfg, st["mean"], st["std"]); net_d.load_state_dict(net.state_dict()); net_d.to(dev).eval()
        bd = {k: v.to(dev) for k, v in b1.items()}
        net_d(bd); sync(dev)                                  # first call compiles device kernels
        t = time.time(); got = net_d(bd)[0]; sync(dev); t_dev = time.time() - t
        got = got.cpu()
    rep["parity_max_abs_diff"] = float((ref - got).abs().max())
    rep["parity_rel"] = float((ref - got).norm() / (ref.norm() + 1e-12))
    rep["forward_s"] = dict(cpu=round(t_cpu, 3), device=round(t_dev, 3))
    ok_parity = rep["parity_rel"] < 1e-3
    _say(f"parity vs CPU: rel {rep['parity_rel']:.2e}, max abs {rep['parity_max_abs_diff']:.2e} "
         f"({'OK' if ok_parity else 'CHECK'}); forward batch 1: cpu {t_cpu:.2f} s, {dev} {t_dev:.2f} s")
    del net, ref
    if precision != "fp32":           # accuracy of the reduced-precision forward against the fp32 device forward
        with torch.no_grad(), autocast_ctx(dev, precision):
            lowp = net_d(bd)[0].float().cpu()
        rep["reduced_precision_rel_diff"] = float((lowp - got).norm() / (got.norm() + 1e-12))
        _say(f"{precision} forward vs fp32: rel {rep['reduced_precision_rel_diff']:.2e}")
    net_d.train()
    opt = torch.optim.AdamW(net_d.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(dev.split(":")[0]) if precision == "fp16" else None
    rep["batches"] = []
    for bs in batch_sizes:
        _say(f"training step, batch {bs} ...")
        try:
            b = {k: v.to(dev) for k, v in batch(bs).items()}
            times = []
            for i in range(n_steps + 1):
                sync(dev); t = time.time()
                with autocast_ctx(dev, precision):
                    pred = net_d(b)[0]
                loss, _ = net_d.loss(pred.float(), b)
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss in {precision}")
                sync(dev); times.append(time.time() - t)
                if times[-1] > max_step_s:
                    break
            dt = float(np.mean(times[1:])) if len(times) > 1 else times[0]
            r = dict(batch_size=bs, s_per_step=round(dt, 3), samples_per_s=round(bs / dt, 2),
                     first_step_s=round(times[0], 2), mem_gb=memory_gb(dev), ok=True)
            rep["batches"].append(r)
            _say(f"batch {bs}: {r['s_per_step']} s/step, {r['samples_per_s']} samples/s, memory {r['mem_gb']}")
            if dt > max_step_s:
                _say(f"step time above {max_step_s} s; stopping the sweep")
                break
        except RuntimeError as e:          # out of memory or unsupported operator
            rep["batches"].append(dict(batch_size=bs, ok=False, error=str(e)[:400]))
            _say(f"batch {bs}: FAILED ({str(e).splitlines()[0][:160]})")
            break
        finally:
            b = None
            empty_cache(dev)
    fits = [r for r in rep["batches"] if r.get("ok")]
    rep["recommended_batch_size"] = fits[-1]["batch_size"] if fits else None
    rep["total_s"] = round(time.time() - t_all, 1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    _say(f"recommended batch_size = {rep['recommended_batch_size']}; report written to {out} ({rep['total_s']} s)")
    return rep
