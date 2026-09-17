"""Turn `Case` shards into fixed-size training samples: context points, token groups, mask, and query points.

Sample contract (keys of `make_sample`)
    presence        (4,)  target presence per channel group: which targets exist for the loss and for metrics.
                          The encoder never reads it, so stored labels cannot change a fully masked prediction.
    ctx_obs         (N,4) observed presence per context point = visible AND the group exists. All zero in the
                          fully masked mode, whether or not the case carries labels.
    prior_presence  (4,)  per-group availability of the prior; context and query priors share it.
    tok_valid       (T,)  token validity; tokens beyond the case's own token count are padding.
    nb_valid        (T,K) neighbour validity; neighbour slots beyond the case's point count are padding.
Samples of any size therefore batch together, and padding never changes an unpadded case's result.

Randomness is keyed: sample (case_index, epoch, draw) always uses the generator seeded by
[seed, case_index, epoch, draw], independent of worker count, process id and restarts.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..data.sampling import MIRROR_Y, context_indices, rotation_z, subsample, transform_case
from ..model.geometry import build_tokens
from ..model.masking import MODES, draw_ratio, point_mask
from ..schema import CHANNEL_GROUP, CLOSURES, GROUNDS, N_FIELDS, PRIOR_KINDS, Case, encode_fields


@dataclass
class SampleSpec:
    n_context: int = 16384
    n_query: int = 4096
    n_tokens_scale1: int = 512
    n_tokens_scale2: int = 256
    k_neighbors: int = 32
    r1: float = 0.05
    r2: float = 0.25
    mask_ratios: Sequence[float] = (0.6, 0.9, 1.0)
    mask_probs: Sequence[float] = (0.5, 0.25, 0.25)
    mask_modes: Sequence[str] = MODES
    mode_probs: Sequence[float] = (0.4, 0.2, 0.2, 0.2)
    fixed_ratio: Optional[float] = None     # 1.0 for supervised regression / fine-tuning / inference
    prior_prob: float = 0.5                 # probability of showing an available prior during pretraining
    use_prior: Optional[bool] = None        # force prior on / off (None = random with prior_prob)
    augment: bool = True                    # master switch for the exact symmetry augmentations below
    rotate: bool = True                     # rotation about z where Case.cond.rotation_ok (protocol: so2_rotation_about_z)
    mirror: bool = True                     # reflection y -> -y (protocol: spanwise_mirror)


def case_to_arrays(case: Case, use_prior: bool) -> dict:
    """Per-point arrays of one (already subsampled) case, in network space (fields encoded, not standardised)."""
    n = case.n
    f = encode_fields(case.fields) if case.fields is not None else np.zeros((n, N_FIELDS), np.float32)
    pres = case.presence.astype(np.float32) if case.fields is not None else np.zeros(4, np.float32)
    has_prior = use_prior and case.prior is not None
    prior = encode_fields(case.prior) if has_prior else np.zeros((n, N_FIELDS), np.float32)
    ppres = case.prior_presence.astype(np.float32) if has_prior else np.zeros(4, np.float32)
    # absent channel groups are exact zeros (selected, not multiplied), whatever the stored array holds there
    f = np.where((pres[CHANNEL_GROUP] > 0)[None, :], f, 0.0).astype(np.float32)
    prior = np.where((ppres[CHANNEL_GROUP] > 0)[None, :], prior, 0.0).astype(np.float32)
    cs = case.cell_size if case.cell_size is not None else np.ones(n, np.float32)
    return dict(pos=case.points.astype(np.float32), wd=case.wall_dist.astype(np.float32),
                nrm=case.normal.astype(np.float32), h=case.height().astype(np.float32),
                s=case.streamwise().astype(np.float32), cs=cs.astype(np.float32), field=f.astype(np.float32),
                prior=prior.astype(np.float32), presence=pres, prior_presence=ppres,
                cs_flag=np.float32(case.cell_size is not None),
                cond_vec=case.cond.scalar_vector(), closure=CLOSURES.index(case.cond.closure),
                ground=GROUNDS.index(case.cond.ground),
                prior_kind=PRIOR_KINDS.index(case.cond.prior_kind) if has_prior else 0)


def validate_case(case: Case) -> None:
    """Reject inputs the model cannot use safely: non-positive or non-finite reference scales, non-finite geometry,
    and non-finite values in stored fields or priors. Multiplying a NaN by a zero mask does not remove it."""
    if not (np.isfinite(case.L_ref) and case.L_ref > 0 and np.isfinite(case.U_ref) and case.U_ref > 0):
        raise ValueError(f"{case.case_id}: L_ref and U_ref must be finite and positive ({case.L_ref}, {case.U_ref})")
    for name in ("points", "wall_dist", "normal", "cell_size"):
        v = getattr(case, name)
        if v is not None and not np.isfinite(v).all():
            raise ValueError(f"{case.case_id}: non-finite values in {name}")
    for name, pres in (("fields", case.presence), ("prior", case.prior_presence)):
        v = getattr(case, name)
        if v is None or pres is None:
            continue
        cols = [c for c, g in enumerate(CHANNEL_GROUP) if pres[g]]
        if cols and not np.isfinite(v[:, cols]).all():
            raise ValueError(f"{case.case_id}: non-finite values in present {name} channels")


def check_query_prior(case: Case, query_case: Case, use_prior: bool) -> None:
    """Context and query must be in one frame (same L_ref, U_ref) and, when a prior is shown, carry the prior on the
    same channel groups; otherwise the shared prior flags would describe the query prior wrongly."""
    if not (np.isclose(case.L_ref, query_case.L_ref) and np.isclose(case.U_ref, query_case.U_ref)):
        raise ValueError(f"query frame differs from the context frame: L_ref {query_case.L_ref} vs {case.L_ref}, "
                         f"U_ref {query_case.U_ref} vs {case.U_ref}")
    if not use_prior or case.prior is None:
        return
    if query_case.prior is None:
        raise ValueError("the context shows a prior but the query points carry none; attach the same prior to both")
    qp = query_case.prior_presence if query_case.prior_presence is not None else np.zeros(4, bool)
    cp = case.prior_presence if case.prior_presence is not None else np.zeros(4, bool)
    if not np.array_equal(np.asarray(qp, bool), np.asarray(cp, bool)):
        raise ValueError(f"query prior groups {np.asarray(qp, bool).tolist()} differ from the context prior groups "
                         f"{np.asarray(cp, bool).tolist()}")


def pad_tokens(tok: dict, n_tokens: int, k: int) -> dict:
    """Pad token arrays to (n_tokens, k) with validity masks. Padded neighbour slots point at row 0 and padded tokens
    at centre 0; both are excluded by the masks, so the values are never used."""
    T, K = tok["nb_idx"].shape
    if T > n_tokens or K > k:
        raise ValueError(f"token arrays ({T}, {K}) exceed the budget ({n_tokens}, {k})")
    nb = np.zeros((n_tokens, k), np.int64); nb[:T, :K] = tok["nb_idx"]
    nb_valid = np.zeros((n_tokens, k), bool); nb_valid[:T, :K] = True
    tok_valid = np.zeros(n_tokens, bool); tok_valid[:T] = True

    def fill(v, pad):
        out = np.full(n_tokens, pad, dtype=v.dtype); out[:T] = v
        return out
    return dict(centre_idx=fill(tok["centre_idx"], 0), nb_idx=nb, scale=fill(tok["scale"], 0),
                radius=fill(tok["radius"], np.float32(1.0)), r_eff=fill(tok["r_eff"], np.float32(1.0)),
                tok_valid=tok_valid, nb_valid=nb_valid)


def select_context(case: Case, n_context: int, rng: np.random.Generator):
    """Geometry-only context selection. Returns (context indices, wake-query indices or None).

    A stratified training shard gives its context from the geometry-defined strata; its wake stratum (a function of
    the solution) only ever serves as extra loss queries. A shard that kept every cell of its case was not
    subsampled, so none of its points is withheld from the context (older shard builds marked a wake stratum there,
    which made context membership depend on the solution). A full-field case is sampled with the same near-wall
    share as the training shards."""
    if case.meta.get("sampling") == "stratified" and case.stratum is not None:
        complete = int(case.meta.get("n_cells", -1)) == case.n
        pool = np.arange(case.n) if complete else np.flatnonzero(case.stratum != 2)
        ctx = pool[np.sort(rng.choice(pool.size, min(n_context, pool.size), replace=False))]
        return ctx, (None if complete else np.flatnonzero(case.stratum == 2))
    if case.n > n_context:
        return context_indices(case.wall_dist, n_context, rng), None
    return np.arange(case.n), None


def make_sample(case: Case, spec: SampleSpec, rng: np.random.Generator, query_case: Optional[Case] = None) -> dict:
    """Build one training / inference sample. If `query_case` is given the queries are its points
    (arbitrary mesh); otherwise queries are drawn from the masked context points."""
    if case.n < 1:
        raise ValueError(f"{case.case_id}: a case needs at least one point")
    if query_case is not None and query_case.n < 1:
        raise ValueError(f"{query_case.case_id}: the query set is empty")
    validate_case(case)
    if query_case is not None:
        validate_case(query_case)
    if spec.augment:
        # one transform for context and explicit queries, so both stay in the same frame
        M = np.eye(3, dtype=np.float32)
        if case.cond.rotation_ok and spec.rotate:
            M = rotation_z(float(rng.uniform(0, 2 * np.pi))) @ M
        if spec.mirror and rng.random() < 0.5:
            M = MIRROR_Y @ M
        if not np.allclose(M, np.eye(3)):
            case = transform_case(case, M)
            if query_case is not None:
                query_case = transform_case(query_case, M)
    ctx, wake = select_context(case, spec.n_context, rng)
    extra_q = subsample(case, wake) if (query_case is None and wake is not None) else None
    case = subsample(case, ctx)
    use_prior = spec.use_prior if spec.use_prior is not None else bool(rng.random() < spec.prior_prob)
    if query_case is not None:
        check_query_prior(case, query_case, use_prior)
    a = case_to_arrays(case, use_prior)
    tok = build_tokens(a["pos"], a["wd"], spec.n_tokens_scale1, spec.n_tokens_scale2, spec.k_neighbors,
                       spec.r1, spec.r2, seed=int(rng.integers(1 << 30)))
    ratio = spec.fixed_ratio if spec.fixed_ratio is not None else draw_ratio(rng, spec.mask_ratios, spec.mask_probs)
    mode = str(rng.choice(spec.mask_modes, p=spec.mode_probs))
    coarse = tok["centre_idx"][tok["scale"] == 1]
    hidden = point_mask(a["pos"], a["wd"], a["s"], coarse, ratio, mode, rng)
    tok = pad_tokens(tok, spec.n_tokens_scale1 + spec.n_tokens_scale2, spec.k_neighbors)
    obs = (~hidden)[:, None] & (a["presence"] > 0)[None, :]
    out = dict(ctx_pos=a["pos"], ctx_wd=a["wd"], ctx_nrm=a["nrm"], ctx_h=a["h"], ctx_s=a["s"], ctx_cs=a["cs"],
               ctx_field=a["field"] * (~hidden)[:, None], ctx_prior=a["prior"], ctx_vis=~hidden, ctx_obs=obs,
               presence=a["presence"], prior_presence=a["prior_presence"], cs_flag=a["cs_flag"],
               cond_vec=a["cond_vec"], closure=a["closure"], ground=a["ground"], prior_kind=a["prior_kind"],
               tok_centre=tok["centre_idx"], tok_nb=tok["nb_idx"], tok_scale=tok["scale"],
               tok_radius=tok["radius"], tok_reff=tok["r_eff"], tok_valid=tok["tok_valid"],
               nb_valid=tok["nb_valid"], mask_ratio=np.float32(ratio), mask_realised=np.float32(hidden.mean()))
    keys = ("pos", "wd", "nrm", "h", "s", "cs", "prior", "field")
    if query_case is None:
        # queries: hidden context points plus solution-stratified wake points (never part of the context)
        if extra_q is not None and extra_q.n > 0:
            ea = case_to_arrays(extra_q, use_prior)
            pool = {k: np.concatenate([a[k][hidden], ea[k]]) for k in keys}
        else:
            pool = {k: a[k][hidden] for k in keys}
        n_pool = pool["pos"].shape[0]
        qi = rng.choice(n_pool, spec.n_query, replace=n_pool < spec.n_query)
        q = {k: pool[k][qi] for k in keys}
        q_loss = np.ones(spec.n_query, bool)
    else:
        qa = case_to_arrays(query_case, use_prior)
        q = {k: qa[k] for k in keys}
        q_loss = np.ones(query_case.n, bool)
    out.update(q_pos=q["pos"], q_wd=q["wd"], q_nrm=q["nrm"], q_h=q["h"], q_s=q["s"], q_cs=q["cs"],
               q_prior=q["prior"], q_field=q["field"], q_loss=q_loss)
    return pad_context(out, spec.n_context)


def pad_context(sample: dict, n: int) -> dict:
    """Zero-pad per-point context arrays to exactly n rows so cases of any size batch together. The encoder reads
    context points only through the valid token groups (tok_centre, tok_nb under tok_valid / nb_valid), which never
    index the padded rows, so the padding has no effect on the output."""
    m = sample["ctx_pos"].shape[0]
    if m >= n:
        return sample
    for k in [k for k in sample if k.startswith("ctx_")]:
        v = sample[k]
        pad = np.zeros((n - m,) + v.shape[1:], dtype=v.dtype)
        sample[k] = np.concatenate([v, pad])
    return sample


def to_torch(sample: dict) -> dict:
    out = {}
    for k, v in sample.items():
        v = np.array(v)                       # keeps 0-d scalars 0-d (ascontiguousarray would promote to 1-d)
        t = torch.from_numpy(v)
        out[k] = t.long() if v.dtype.kind in "iu" else (t.bool() if v.dtype == bool else t.float())
    return out


SAMPLER_VERSION = "keyed-permutation-1"


class ShardDataset(Dataset):
    """Map-style dataset over every shard in `shard_paths`; its length is the pool size.

    An item is addressed by a sample key (case_index, epoch, draw). The key alone determines the sample: the
    generator is seeded with [seed, case_index, epoch, draw]. A plain integer i is the fixed key (i, 0, 0), which is
    what validation uses. Keys come from `KeyedSampler` in the main process, so persistent workers see new epochs."""

    def __init__(self, shard_paths: Sequence[str], spec: SampleSpec, seed: int = 0):
        self.paths = list(shard_paths); self.spec = spec; self.seed = int(seed)
        if not self.paths:
            raise ValueError("empty shard list")
        self.numpy_out = False

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def key(i) -> tuple:
        if isinstance(i, (tuple, list)):
            return int(i[0]), int(i[1]), int(i[2])
        return int(i), 0, 0

    def sample(self, key) -> dict:
        ci, ep, dr = self.key(key)
        rng = np.random.default_rng([self.seed, ci, ep, dr])
        s = make_sample(Case.load(self.paths[ci]), self.spec, rng)
        s["sample_key"] = np.array([ci, ep, dr], np.int64)
        return s

    def __getitem__(self, i):
        s = self.sample(i)
        return {k: np.array(v) for k, v in s.items()} if self.numpy_out else to_torch(s)


class KeyedSampler(Sampler):
    """Epoch-wise sample keys over the whole pool. Each epoch is a seeded permutation, so one epoch visits every case
    exactly once and eligibility does not depend on the batch size. With `groups` (one label per case) and
    `weights` (label -> relative share), an epoch instead draws round(len(pool) * share) keys per group by cycling
    through fresh seeded permutations of that group, then shuffles the union.

    `set_epoch(epoch, start)` positions the sampler; `start` is the number of keys of that epoch already consumed,
    which makes a resumed run continue the same key stream."""

    def __init__(self, n: int, seed: int = 0, groups: Optional[Sequence[str]] = None,
                 weights: Optional[dict] = None):
        self.n, self.seed = int(n), int(seed)
        self.groups = None if groups is None else np.asarray(groups)
        self.weights = weights
        if weights is not None:
            if self.groups is None or len(self.groups) != self.n:
                raise ValueError("weights need one group label per case")
            unknown = set(weights) - set(self.groups.tolist())
            missing = set(self.groups.tolist()) - set(weights)
            if unknown or missing:
                raise ValueError(f"sampling weights do not match the pool groups: unknown {sorted(unknown)}, "
                                 f"missing {sorted(missing)}")
            if any(float(w) < 0 for w in weights.values()) or sum(float(w) for w in weights.values()) <= 0:
                raise ValueError("sampling weights must be non-negative with a positive sum")
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = int(epoch), int(start)

    def epoch_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, 0x5A4D, int(epoch)])
        if self.weights is None:
            return rng.permutation(self.n)
        tot = sum(float(w) for w in self.weights.values())
        parts = []
        for g in sorted(self.weights):
            idx = np.flatnonzero(self.groups == g)
            m = int(round(self.n * float(self.weights[g]) / tot))
            reps = [rng.permutation(idx) for _ in range(-(-m // max(idx.size, 1)))] if m else []
            parts.append(np.concatenate(reps)[:m] if reps else np.zeros(0, np.int64))
        order = np.concatenate(parts)
        return order[rng.permutation(order.size)]

    def epoch_length(self) -> int:
        return int(self.epoch_order(0).size) if self.weights is not None else self.n

    def __iter__(self) -> Iterator[tuple]:
        order = self.epoch_order(self.epoch)
        for pos in range(self.start, order.size):
            yield int(order[pos]), self.epoch, pos

    def __len__(self):
        return max(0, self.epoch_length() - self.start)


def read_manifest(manifest_path: str):
    import pandas as pd
    return pd.read_parquet(manifest_path)


def shard_paths(manifest_path: str, split: str, tiers: Optional[Sequence[str]] = None,
                source: Optional[str] = None) -> list:
    df = read_manifest(manifest_path)
    df = df[df["split"] == split]
    if tiers:
        df = df[df["tier"].isin(tiers)]
    if source:
        df = df[df["source"] == source]
    return sorted(df["shard"].tolist())


POOL_COLUMNS = ("case_id", "source", "split", "shard")
NATIVE_SOURCES = ("aero_urban",)       # sources whose conversion is defined by this package's own adapter


def load_pool(manifests: Sequence[str], sources: Optional[Sequence[str]] = None,
              exclude_sources: Sequence[str] = ()):
    """Combine explicit per-source manifests into one validated pool (a DataFrame).

    Fails closed on: a missing manifest or column, a requested source absent from every manifest, duplicate case
    ids, a case listed in more than one split, a family id shared across splits (when a `family` column exists),
    and any case of an excluded (held-out) source in the pool."""
    import pandas as pd
    if not manifests:
        raise ValueError("no manifests given")
    frames = []
    for m in manifests:
        if not os.path.exists(m):
            raise FileNotFoundError(f"manifest not found: {m}")
        df = pd.read_parquet(m)
        miss = [c for c in POOL_COLUMNS if c not in df.columns]
        if miss:
            raise ValueError(f"{m}: missing columns {miss}")
        if "error" in df.columns:
            df = df[df["error"].isna()]
        frames.append(df.assign(manifest=os.path.abspath(m)))
    pool = pd.concat(frames, ignore_index=True)
    if sources is not None:
        absent = sorted(set(sources) - set(pool["source"]))
        if absent:
            raise ValueError(f"requested sources not found in any manifest: {absent}")
        pool = pool[pool["source"].isin(list(sources))]
    bad = sorted(set(pool["source"]) & set(exclude_sources))
    if bad:
        raise ValueError(f"held-out sources present in the pretraining pool: {bad}")
    ext = pool[~pool["source"].isin(NATIVE_SOURCES)]
    if len(ext):                              # external sources need a validated conversion per case
        ok = ext["validated"].fillna(False).astype(bool) if "validated" in ext.columns else None
        if ok is None or not ok.all():
            srcs = sorted(ext["source"].unique()) if ok is None else sorted(ext.loc[~ok, "source"].unique())
            raise ValueError(f"cases of sources {srcs} are not marked validated (manifest column `validated`)")
    dup = pool["case_id"][pool["case_id"].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{len(dup)} case ids appear more than once, e.g. {dup[:3]}")
    if "family" in pool.columns:
        fam = pool.dropna(subset=["family"]).groupby("family")["split"].nunique()
        shared = fam[fam > 1].index.tolist()
        if shared:
            raise ValueError(f"{len(shared)} families span more than one split, e.g. {shared[:3]}")
    return pool.sort_values("shard").reset_index(drop=True)


def pool_fingerprint(pool) -> str:
    """sha256 over the resolved (case_id, source, split, shard) rows."""
    rows = pool[list(POOL_COLUMNS)].astype(str).sort_values("case_id").to_csv(index=False)
    return hashlib.sha256(rows.encode()).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


VELOCITY_SCALES = ("per_channel", "common")


def channel_stats(paths: Sequence[str], n_per_case: int = 4096, seed: int = 0, rotation_invariant: bool = True,
                  velocity_scale: str = "per_channel") -> dict:
    """Per-channel mean / std in network space over present channels (training split only).

    `velocity_scale`: "per_channel" standardises Uz with its own standard deviation (the loss then weights a
    physical Uz error by (sigma_h / sigma_z)^2 relative to a horizontal error); "common" gives all three velocity
    components the horizontal RMS, so the velocity part of the loss is isotropic in physical velocity error. Which
    one to train with is an objective-design choice to be settled by a matched experiment; the choice is recorded.
    """
    if velocity_scale not in VELOCITY_SCALES:
        raise ValueError(f"velocity_scale must be one of {VELOCITY_SCALES}")
    rng = np.random.default_rng(seed)
    acc = [[] for _ in range(N_FIELDS)]
    n_neg = np.zeros(2, np.int64); n_tot = np.zeros(2, np.int64); n_bad = 0
    for p in paths:
        c = Case.load(p)
        idx = rng.choice(c.n, min(n_per_case, c.n), replace=False)
        raw = c.fields[idx]
        n_bad += int((~np.isfinite(raw)).sum())
        for j, ch in enumerate((4, 5)):                   # negative k / eps are clipped by encode_fields
            if c.presence[ch - 2]:
                n_neg[j] += int((raw[:, ch] < 0).sum()); n_tot[j] += len(idx)
        f = encode_fields(raw)
        for ch, g in enumerate([0, 0, 0, 1, 2, 3]):
            if c.presence[g]:
                acc[ch].append(f[:, ch])
    mean = [float(np.concatenate(a).mean()) if a else 0.0 for a in acc]
    std = [float(np.concatenate(a).std()) + 1e-6 if a else 1.0 for a in acc]
    if rotation_invariant and acc[0] and acc[1]:
        # rotation about z mixes Ux and Uy: standardise both with zero mean and the horizontal RMS
        rms = float(np.sqrt(0.5 * (np.mean(np.concatenate(acc[0]) ** 2) + np.mean(np.concatenate(acc[1]) ** 2))))
        mean[0] = mean[1] = 0.0; std[0] = std[1] = rms
        if velocity_scale == "common":
            mean[2] = 0.0; std[2] = rms
    elif velocity_scale == "common" and acc[0] and acc[1] and acc[2]:
        rms = float(np.sqrt(np.mean(np.concatenate(acc[0] + acc[1] + acc[2]) ** 2)))
        mean[0] = mean[1] = mean[2] = 0.0; std[0] = std[1] = std[2] = rms
    if not all(np.isfinite(mean)) or not all(np.isfinite(std)) or min(std) <= 0:
        raise ValueError("channel statistics are not finite and positive")
    return dict(mean=mean, std=std, n_cases=len(paths), n_per_case=n_per_case, rotation_invariant=rotation_invariant,
                velocity_scale=velocity_scale,
                clipped_fraction=dict(k=float(n_neg[0] / max(n_tot[0], 1)), eps=float(n_neg[1] / max(n_tot[1], 1))),
                non_finite_values=n_bad)


def tokeniser_stats(paths: Sequence[str], spec: SampleSpec, n_cases: int = 8, seed: int = 0) -> dict:
    """Coverage and effective radii of the token groups on real shards, with the training context selection.

    Per case: the geometry-only context of `spec.n_context` points, `build_tokens` with the spec's budget, then the
    fraction of context points inside at least one group (overall and for the near-wall band) and the quantiles of
    the k-th-neighbour distance per scale (index 0 wall-weighted, 1 uniform). Medians over the cases are reported
    next to the nominal radii, so the paper can state the realised support of each scale."""
    from ..model.geometry import build_tokens, coverage
    rng = np.random.default_rng(seed)
    rows = []
    for p in list(paths)[:n_cases]:
        c = Case.load(p)
        ctx, _ = select_context(c, spec.n_context, rng)
        c = subsample(c, ctx)
        tok = build_tokens(c.points, c.wall_dist, spec.n_tokens_scale1, spec.n_tokens_scale2, spec.k_neighbors,
                           spec.r1, spec.r2, seed=int(rng.integers(1 << 30)))
        near = c.wall_dist < 0.25
        seen = np.zeros(c.n, bool); seen[tok["nb_idx"].ravel()] = True
        r = dict(case_id=c.case_id, n_context=int(c.n), coverage=coverage(c.n, tok["nb_idx"]),
                 coverage_near_wall=float(seen[near].mean()) if near.any() else float("nan"),
                 coverage_far=float(seen[~near].mean()) if (~near).any() else float("nan"))
        for sc in (0, 1):
            re = tok["r_eff"][tok["scale"] == sc]
            r[f"r_eff_scale{sc}_q10"], r[f"r_eff_scale{sc}_q50"], r[f"r_eff_scale{sc}_q90"] = \
                [float(x) for x in np.quantile(re, [0.1, 0.5, 0.9])]
        rows.append(r)
    keys = [k for k in rows[0] if k != "case_id"]
    return dict(n_cases=len(rows), nominal_r1=spec.r1, nominal_r2=spec.r2, k_neighbors=spec.k_neighbors,
                median={k: float(np.nanmedian([r[k] for r in rows])) for k in keys}, cases=rows)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def numpy_collate(samples: list) -> dict:
    """Stack numpy samples; tensors are created in the main process, so workers need no shared memory."""
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}
