"""CFD2vec network: point-set tokeniser -> pre-LN Transformer with 3-D rotary attention -> implicit field decoder.

Input modalities
    Geometry    wall distance, wall normal, height above ground, body-relative streamwise coordinate
    Mesh        point locations and local cell size (the representation never assumes a mesh type)
    BC          inflow direction, profile exponent, roughness, turbulence intensity, ground type
    Physics     Reynolds number, turbulence closure / fidelity (token dropout so it may be omitted)
    Fields      visible U, Cp, k, eps values (masked field modelling) and an optional low-fidelity prior
Raw absolute coordinates are never encoded. Positions enter through rotary attention and through physically
meaningful coordinates. Attention between spatial tokens depends only on their relative position. The conditioning
and CLS tokens sit at position zero, the canonical body origin, so attention between a spatial token and a
conditioning token depends on the spatial token's position relative to that origin (a body-relative, not an
origin-free, encoding).

Input schemas (ModelConfig.input_schema; checkpoints without the field are schema 1)
    1  legacy: the encoder sees the case's target presence on every point, and the prior as values plus one
       any-group flag. A fully masked prediction then depends on whether the case carries labels.
    2  current: the encoder sees per-point observed presence (visible AND present; all zero when fully masked) and
       all four prior group flags, in the point and query features. Target presence is used only by the loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..schema import CHANNEL_GROUP, CLOSURES, GROUNDS, N_FIELDS, PRIOR_KINDS

N_COND = 12
LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5)
INPUT_SCHEMA = 2                  # schema of newly built models; see the module docstring
LEGACY_SCHEMA = 1                 # assumed for checkpoints that do not record one


@dataclass
class ModelConfig:
    name: str = "cfd2vec-pilot"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    n_tokens_scale1: int = 512
    n_tokens_scale2: int = 256
    r1: float = 0.05
    r2: float = 0.25
    k_neighbors: int = 32
    pointnet_hidden: list = field(default_factory=lambda: [64, 128])
    n_cross_layers: int = 2
    closure_dropout: float = 0.3
    min_wavelength: float = 0.02
    max_wavelength: float = 50.0
    n_context: int = 16384
    n_query: int = 4096
    grad_checkpoint: bool = False     # recompute block activations in backward: less memory, ~30 % more compute
    decode_chunk: int = 0             # query points per decoder chunk during training (0 = all at once)
    input_schema: int = INPUT_SCHEMA  # feature layout of the encoder / decoder inputs

    def __post_init__(self):
        if self.input_schema not in (1, 2):
            raise ValueError(f"unknown input_schema {self.input_schema}")
        for k in ("d_model", "n_layers", "n_heads", "n_tokens_scale1", "n_tokens_scale2", "k_neighbors",
                  "n_context", "n_query"):
            if int(getattr(self, k)) < 1:
                raise ValueError(f"{k} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not (self.r1 > 0 and self.r2 > 0 and 0 < self.min_wavelength < self.max_wavelength):
            raise ValueError("r1, r2 and the rotary wavelengths must be positive, min < max")

    @classmethod
    def from_checkpoint(cls, d: dict) -> "ModelConfig":
        """Config stored in a checkpoint; checkpoints written before schemas were versioned are schema 1."""
        d = dict(d)
        d.setdefault("input_schema", LEGACY_SCHEMA)
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        import yaml
        y = yaml.safe_load(open(path))
        flat = {k: v for k, v in y.items() if not isinstance(v, dict)}
        for sec in ("tokenizer", "decoder", "cond", "rope", "data", "memory"):
            flat.update(y.get(sec, {}))
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in flat.items() if k in known})

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------------------------- rotary attention
class Rope3D(nn.Module):
    """Rotary position embedding on 3-D coordinates: each axis rotates its own block of channel pairs, with
    wavelengths log-spaced between min_wavelength and max_wavelength (L_ref units). Attention scores then depend
    only on relative positions between the rotated tokens."""

    def __init__(self, head_dim: int, min_wl: float, max_wl: float):
        super().__init__()
        self.nf = head_dim // 6
        wl = torch.logspace(math.log10(min_wl), math.log10(max_wl), self.nf)
        self.register_buffer("omega", 2 * math.pi / wl, persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x (B,h,T,hd), pos (B,T,3)
        if self.nf == 0:
            return x
        ang = pos[:, None, :, :, None] * self.omega            # (B,1,T,3,nf)
        ang = ang.flatten(-2)                                   # (B,1,T,3nf)
        cos, sin = ang.cos(), ang.sin()
        n = 6 * self.nf
        xr, xp = x[..., :n], x[..., n:]
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
        return torch.cat([out, xp], -1)


class Attention(nn.Module):
    def __init__(self, d: int, h: int, rope: Rope3D, dropout: float = 0.0):
        super().__init__()
        self.h, self.hd = h, d // h
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.rope, self.dropout = rope, dropout

    def forward(self, x, pos_x, mem=None, pos_m=None, key_mask=None):
        """key_mask (B, Tk) bool, True for keys that take part; None = all keys."""
        mem = x if mem is None else mem
        pos_m = pos_x if pos_m is None else pos_m
        B, Tq, D = x.shape
        q = self.q(x).view(B, Tq, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.shape[1], 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, pos_x), self.rope(k, pos_m)
        am = None if key_mask is None else key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, Tq, D))


class Block(nn.Module):
    def __init__(self, d, h, mlp_ratio, rope, dropout=0.0, cross=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.nm = nn.LayerNorm(d) if cross else None
        self.attn = Attention(d, h, rope, dropout)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x, pos_x, mem=None, pos_m=None, key_mask=None):
        if self.nm is None:
            x = x + self.attn(self.n1(x), pos_x, key_mask=key_mask)
        else:
            x = x + self.attn(self.n1(x), pos_x, self.nm(mem), pos_m, key_mask=key_mask)
        return x + self.mlp(self.n2(x))


def mlp(sizes, act=nn.GELU):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


# ----------------------------------------------------------------------------------------------- features
def geometry_features(wd, nrm, h, s, cs, cs_flag):
    """Per-point geometry + mesh features (B,N,9): log / clipped wall distance, normal (3), height, streamwise
    coordinate, log cell size and its presence flag."""
    f = cs_flag.reshape(-1, 1, 1).to(wd.dtype).expand(wd.shape[0], wd.shape[1], 1)
    return torch.cat([torch.log(wd.clamp(min=0) + 1e-2)[..., None], wd.clamp(max=5.0)[..., None], nrm,
                      h.clamp(-5, 20)[..., None] / 5.0, (s / 5.0).clamp(-10, 10)[..., None],
                      torch.log(cs.clamp(min=1e-4))[..., None] * f / 5.0, f], -1)


N_GEOM = 9


def n_prior_flags(schema: int) -> int:
    return 1 if schema == 1 else 4


def n_point_feat(schema: int) -> int:
    """rel pos (3), geometry, fields, visibility, presence (4; schema 2: observed), prior, prior flags."""
    return 3 + N_GEOM + N_FIELDS + 1 + 4 + N_FIELDS + n_prior_flags(schema)


def n_query_feat(schema: int) -> int:
    return N_GEOM + N_FIELDS + n_prior_flags(schema)


N_POINT_FEAT = n_point_feat(LEGACY_SCHEMA)       # legacy constant; use n_point_feat(schema)


def _all_true(m) -> bool:
    return m is None or bool(m.all())


class CFD2vecNet(nn.Module):
    def __init__(self, cfg: ModelConfig, channel_mean=None, channel_std=None):
        super().__init__()
        self.cfg = cfg
        self.schema = int(cfg.input_schema)
        d, h = cfg.d_model, cfg.n_heads
        self.register_buffer("ch_mean", torch.zeros(N_FIELDS) if channel_mean is None else torch.as_tensor(channel_mean, dtype=torch.float32))
        self.register_buffer("ch_std", torch.ones(N_FIELDS) if channel_std is None else torch.as_tensor(channel_std, dtype=torch.float32))
        if not (torch.isfinite(self.ch_mean).all() and torch.isfinite(self.ch_std).all() and (self.ch_std > 0).all()):
            raise ValueError("channel statistics must be finite with positive standard deviations")
        self.register_buffer("ch_group", torch.as_tensor(CHANNEL_GROUP, dtype=torch.long), persistent=False)
        self.register_buffer("loss_w", torch.tensor(LOSS_WEIGHTS), persistent=False)
        hid = list(cfg.pointnet_hidden)
        self.pointnet = mlp([n_point_feat(self.schema)] + hid)
        self.tok_proj = nn.Linear(hid[-1], d)
        self.tok_geom = mlp([N_GEOM + 2, d, d])                       # centre geometry + log radius + log r_eff
        self.scale_emb = nn.Embedding(2, d)
        self.cond_mlp = mlp([N_COND, d, d])
        self.closure_emb = nn.Embedding(len(CLOSURES), d)
        self.ground_emb = nn.Embedding(len(GROUNDS), d)
        self.prior_emb = nn.Embedding(len(PRIOR_KINDS), d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
        self.rope = Rope3D(d // h, cfg.min_wavelength, cfg.max_wavelength)
        self.blocks = nn.ModuleList([Block(d, h, cfg.mlp_ratio, self.rope, cfg.dropout) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.q_mlp = mlp([n_query_feat(self.schema), d, d])
        self.dec = nn.ModuleList([Block(d, h, cfg.mlp_ratio, self.rope, cfg.dropout, cross=True)
                                  for _ in range(cfg.n_cross_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_FIELDS)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # starts at the prior (or the mean field)
        self.n_cond_tokens = 5

    # ------------------------------------------------------------------ helpers
    def _ckpt(self) -> bool:
        return bool(self.cfg.grad_checkpoint) and self.training and torch.is_grad_enabled()

    def standardise(self, x):
        return (x - self.ch_mean) / self.ch_std

    def unstandardise(self, x):
        return x * self.ch_std + self.ch_mean

    def channel_presence(self, group_presence):         # (..., 4) -> (..., 6)
        return group_presence[..., self.ch_group]

    def prior_flags(self, b: dict, n: int) -> torch.Tensor:
        """Per-point prior availability features (B, n, 1) in schema 1, (B, n, 4) in schema 2."""
        pp = b["prior_presence"].float()
        B = pp.shape[0]
        if self.schema == 1:
            return (pp > 0).any(-1).float()[:, None, None].expand(B, n, 1)
        return pp[:, None, :].expand(B, n, 4)

    @staticmethod
    def token_masks(b: dict):
        """(tok_valid, nb_valid) or None where every entry is valid (the common full-size case)."""
        tv, nv = b.get("tok_valid"), b.get("nb_valid")
        return (None if _all_true(tv) else tv.bool()), (None if _all_true(nv) else nv.bool())

    # ------------------------------------------------------------------ encoder
    def encode(self, b: dict):
        """b: batch dict (see cfd2vec.train.dataset). Returns tokens (B,T+5,d), positions (B,T+5,3), global (B,2d)."""
        cfg = self.cfg
        B, N, _ = b["ctx_pos"].shape
        geo = geometry_features(b["ctx_wd"], b["ctx_nrm"], b["ctx_h"], b["ctx_s"], b["ctx_cs"], b["cs_flag"])
        vis = b["ctx_vis"].float()[..., None]
        if self.schema == 1:
            chp = self.channel_presence(b["presence"])                                      # (B,6)
            fld = self.standardise(b["ctx_field"]) * vis * chp[:, None]
            pres = b["presence"].float()[:, None].expand(B, N, 4)
        else:
            pres = b["ctx_obs"].float()                                                     # (B,N,4) observed
            fld = self.standardise(b["ctx_field"]) * self.channel_presence(pres)
        pchp = self.channel_presence(b["prior_presence"].float())
        prior = self.standardise(b["ctx_prior"]) * pchp[:, None]
        feat = torch.cat([geo, fld, vis, pres, prior, self.prior_flags(b, N)], -1)          # (B,N,F-3)
        tv, nv = self.token_masks(b)
        nb = b["tok_nb"]                                                                     # (B,T,K)
        T, K = nb.shape[1], nb.shape[2]
        bidx = torch.arange(B, device=nb.device)[:, None, None]
        cpos = torch.gather(b["ctx_pos"], 1, b["tok_centre"][..., None].expand(B, T, 3))    # (B,T,3)
        rel = (b["ctx_pos"][bidx, nb] - cpos[:, :, None]) / b["tok_radius"][..., None, None]
        g = torch.cat([rel, feat[bidx, nb]], -1)                                             # (B,T,K,F)

        def pool(t):
            h = self.pointnet(t)
            if nv is None:
                return h.max(dim=2).values
            h = h.masked_fill(~nv[..., None], float("-inf")).max(dim=2).values
            return h.masked_fill(~nv.any(-1)[..., None], 0.0)          # tokens without neighbours are padding
        tok = self.tok_proj(checkpoint(pool, g, use_reentrant=False) if self._ckpt() else pool(g))
        cgeo = geo[torch.arange(B, device=nb.device)[:, None], b["tok_centre"]]
        tok = tok + self.tok_geom(torch.cat([cgeo, torch.log(b["tok_radius"])[..., None],
                                             torch.log(b["tok_reff"].clamp(min=1e-4))[..., None]], -1))
        tok = tok + self.scale_emb(b["tok_scale"])
        closure = b["closure"]
        if self.training and cfg.closure_dropout > 0:
            drop = torch.rand(B, device=closure.device) < cfg.closure_dropout
            closure = torch.where(drop, torch.full_like(closure, CLOSURES.index("unknown")), closure)
        ctok = torch.stack([self.cond_mlp(b["cond_vec"]), self.closure_emb(closure), self.ground_emb(b["ground"]),
                            self.prior_emb(b["prior_kind"]), self.cls.expand(B, -1, -1)[:, 0]], 1)
        x = torch.cat([ctok, tok], 1)
        pos = torch.cat([torch.zeros(B, self.n_cond_tokens, 3, device=x.device), cpos], 1)
        km = self.key_mask(tv, B, x.device)
        for blk in self.blocks:
            x = checkpoint(blk, x, pos, None, None, km, use_reentrant=False) if self._ckpt() else \
                blk(x, pos, key_mask=km)
        x = self.norm(x)
        sp = x[:, self.n_cond_tokens:]
        if tv is None:
            mean_tok = sp.mean(1)
        else:
            w = tv.float()[..., None]
            mean_tok = (sp * w).sum(1) / w.sum(1).clamp(min=1.0)
        glob = torch.cat([x[:, self.n_cond_tokens - 1], mean_tok], -1)
        return x, pos, glob

    def key_mask(self, tok_valid, B, device):
        """(B, 5 + T) key mask with the conditioning tokens always valid; None when every token is valid."""
        if tok_valid is None:
            return None
        return torch.cat([torch.ones(B, self.n_cond_tokens, dtype=torch.bool, device=device), tok_valid], 1)

    # ------------------------------------------------------------------ decoder
    def decode(self, mem, mem_pos, b: dict, chunk: int = 0):
        """Evaluate fields at query points. Returns standardised, encoded predictions (B,Nq,6)."""
        B, Nq, _ = b["q_pos"].shape
        pchp = self.channel_presence(b["prior_presence"].float())
        km = self.key_mask(self.token_masks(b)[0], B, mem.device)
        out = []
        step = Nq if chunk <= 0 else chunk
        for s in range(0, Nq, step):
            sl = slice(s, s + step)
            geo = geometry_features(b["q_wd"][:, sl], b["q_nrm"][:, sl], b["q_h"][:, sl], b["q_s"][:, sl],
                                    b["q_cs"][:, sl], b["cs_flag"])
            prior = self.standardise(b["q_prior"][:, sl]) * pchp[:, None]
            q = self.q_mlp(torch.cat([geo, prior, self.prior_flags(b, geo.shape[1])], -1))
            for blk in self.dec:
                qp = b["q_pos"][:, sl]
                q = checkpoint(blk, q, qp, mem, mem_pos, km, use_reentrant=False) if self._ckpt() else \
                    blk(q, qp, mem, mem_pos, key_mask=km)
            out.append(self.head(self.dec_norm(q)) + prior)      # residual on the prior when one is given
        return torch.cat(out, 1)

    def forward(self, b: dict, chunk: int | None = None):
        mem, pos, glob = self.encode(b)
        if chunk is None:
            chunk = self.cfg.decode_chunk if self.training else 0
        return self.decode(mem, pos, b, chunk=chunk), glob

    def loss(self, pred, b: dict):
        """Weighted MSE on present channels at masked query points (standardised encoded space). `presence` here is the
        target presence; it selects loss terms and is never an encoder input in schema 2."""
        tgt = self.standardise(b["q_field"])
        m = self.channel_presence(b["presence"])[:, None, :] * b["q_loss"].float()[..., None]    # (B,Nq,6)
        w = m * self.loss_w
        se = (pred - tgt) ** 2
        loss = (se * w).sum() / w.sum().clamp(min=1.0)
        per_ch = (se * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1.0)
        return loss, per_ch.detach()


def migrate_state_dict(sd: dict, from_schema: int, to_schema: int) -> dict:
    """Map schema-1 weights onto a schema-2 network for initialisation (never for resuming a run).

    The single any-group prior flag column is moved to the U-group flag column and the other three flag columns start
    at zero, so a prior with every group present gives the same first-layer activation as before. The presence
    columns keep their weights but now receive observed presence instead of target presence; the result is a
    starting point for training under schema 2, not an equivalent model."""
    if from_schema == to_schema:
        return sd
    if (from_schema, to_schema) != (1, 2):
        raise ValueError(f"no migration from input schema {from_schema} to {to_schema}")
    sd = dict(sd)
    for key in ("pointnet.0.weight", "q_mlp.0.weight"):
        w = sd[key]
        sd[key] = torch.cat([w, torch.zeros(w.shape[0], 3, dtype=w.dtype)], 1)
    return sd


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
