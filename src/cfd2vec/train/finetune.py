"""Few-shot fine-tuning under the frozen protocol (default: the copy shipped in cfd2vec/protocols/).

The same routine serves CFD2vec (pretrained init), the from-scratch control (random init) and the supervised
control (supervised-pretrained init), so the three differ only in their starting weights.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from ..data.sampling import subsample
from ..eval.metrics import rel_l2
from ..schema import Case
from ..tasks.predict import predict_case, spec_from_config
from .dataset import make_sample, to_torch

DECODER_PREFIXES = ("q_mlp", "dec", "dec_norm", "head")


def protocol_lr_factor(step: int, total: int, warmup: int = 0, min_ratio: float = 0.0) -> float:
    """Learning-rate multiplier of the protocol schedule: linear warm-up over `warmup` steps, then cosine from 1
    to `min_ratio` over the remaining steps (the frozen protocol v2 uses warmup 0 and min_ratio 0)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def param_groups(model, lr_enc, lr_dec, wd):
    enc, dec = [], []
    for n, p in model.named_parameters():
        (dec if n.split(".")[0] in DECODER_PREFIXES else enc).append(p)
    return [dict(params=enc, lr=lr_enc, base_lr=lr_enc, weight_decay=wd),
            dict(params=dec, lr=lr_dec, base_lr=lr_dec, weight_decay=wd)]


def val_score(model, cases, device, use_prior, n_points=4096, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for c in cases:
        q = subsample(c, np.sort(rng.choice(c.n, min(n_points, c.n), replace=False)))
        m = rel_l2(predict_case(model, c, query=q, use_prior=use_prior, device=device)["fields"], q.fields, c.presence)
        vals.append(np.mean([m[k] for k in ("rel_l2_U", "rel_l2_Cp", "rel_l2_k") if k in m]))
    return float(np.mean(vals))


def finetune(model, case_paths: Sequence[str], protocol: Optional[str] = None, seed: int = 0, use_prior: bool = False,
             device: str = "cpu", batch_size: int = 4, log_path: Optional[str] = None,
             max_epochs: Optional[int] = None):
    from .dataset import file_sha256
    if protocol is None:
        from ..api import DEFAULT_PROTOCOL
        protocol = DEFAULT_PROTOCOL
    P = yaml.safe_load(open(protocol))
    proto_meta = dict(protocol_path=os.path.abspath(protocol), protocol_version=P["version"],
                      protocol_sha256=file_sha256(protocol))
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    cases = [Case.load(p) for p in case_paths]
    N = len(cases)
    order = rng.permutation(N)
    if N >= P["epochs"]["small_n_threshold"]:
        n_val = max(1, int(round(P["early_stopping"]["val_fraction"] * N)))
        val = [cases[i] for i in order[:n_val]]; train = [cases[i] for i in order[n_val:]]
        epochs = P["epochs"]["max"]; patience = P["early_stopping"]["patience_epochs"]
    else:
        val, train = [], cases
        epochs = P["epochs"]["fixed_epochs_small_n"]; patience = None
    epochs = min(epochs, max_epochs) if max_epochs else epochs
    o = P["optimizer"]
    opt = torch.optim.AdamW(param_groups(model, o["lr_encoder"], o["lr_decoder"], o["weight_decay"]),
                            betas=tuple(o["betas"]))
    aug = P.get("augmentation", {})
    rot, mir = bool(aug.get("so2_rotation_about_z", True)), bool(aug.get("spanwise_mirror", True))
    spec = spec_from_config(model.cfg, fixed_ratio=P.get("mask_ratio", 1.0), use_prior=use_prior,
                            augment=rot or mir, rotate=rot, mirror=mir)
    bs = min(batch_size, len(train))
    steps_per_epoch = math.ceil(len(train) / bs); total = epochs * steps_per_epoch
    sch = P.get("schedule", {})
    if sch.get("name", "cosine") != "cosine":
        raise ValueError(f"protocol schedule {sch.get('name')!r} is not implemented (cosine only)")
    warmup = int(round(float(sch.get("warmup_epochs", 0)) * steps_per_epoch))
    min_ratio = float(sch.get("min_lr_ratio", 0.0))
    best, best_state, bad, hist, step = float("inf"), None, 0, [], 0
    t0 = time.time()
    if val:                                   # the starting weights are a candidate: fine-tuning never returns worse
        best = val_score(model, val, device, use_prior, seed=seed)
        best_state = copy.deepcopy(model.state_dict())
        hist.append(dict(epoch=0, val=best, t_min=0.0))
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train)); losses = []
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            ss = [to_torch(make_sample(train[i], spec, rng)) for i in idx]
            b = {k: torch.stack([x[k] for x in ss]).to(device) for k in ss[0]}
            f = protocol_lr_factor(step, total, warmup, min_ratio)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            loss, _ = model.loss(model(b)[0], b)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip_norm"]); opt.step()
            losses.append(float(loss.detach())); step += 1
        rec = dict(epoch=ep + 1, loss=float(np.mean(losses)), t_min=(time.time() - t0) / 60)
        if val and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            rec["val"] = val_score(model, val, device, use_prior, seed=seed)
            if rec["val"] < best:
                best, best_state, bad = rec["val"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 5
            if patience is not None and bad >= patience:
                hist.append(rec); break
        hist.append(rec)
    if best_state is not None:
        model.load_state_dict(best_state)
    if log_path:
        json.dump(dict(history=hist, N=N, n_train=len(train), n_val=len(val), seed=seed, best_val=best,
                       input_schema=int(getattr(model, "schema", 1)), **proto_meta), open(log_path, "w"), indent=1)
    return model, hist


def finetune_cli(a):
    from ..api import CFD2vec
    m = CFD2vec.from_pretrained(a.init, device=a.device) if a.init else \
        CFD2vec.from_config(a.model_config, a.stats, device=a.device)
    os.makedirs(a.out, exist_ok=True)
    finetune(m.net, a.cases, a.protocol, seed=a.seed, use_prior=a.use_prior, device=m.device,
             log_path=os.path.join(a.out, "finetune.json"))
    m.save(os.path.join(a.out, "model.pt"))
