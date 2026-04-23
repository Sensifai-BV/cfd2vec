"""Public Python API.

    m = CFD2vec.from_pretrained("runs/pilot_masked/best.pt")
    case = cfd2vec.solvers.get_adapter("openfoam").read_case(case_dir, U_ref, None, cond)
    emb = m.encode(case)["embedding"]              # frozen representation
    res = m.predict(case)                          # normalised fields on every cell
    phys = m.to_physical(res["fields"], case)      # U [m/s], p [m^2/s^2], k, epsilon, omega
    m.finetune(shard_paths)                        # few-shot adaptation under the frozen protocol
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import torch

from .model.network import CFD2vecNet, ModelConfig
from .schema import Case
from .train.pretrain import pick_device


def protocol_path(name: str = "finetune_protocol.yaml") -> str:
    """Path of a protocol file shipped inside the package (works from a source checkout and an installed wheel)."""
    from importlib.resources import files
    p = files("cfd2vec").joinpath("protocols", name)
    if not p.is_file():
        raise FileNotFoundError(f"packaged protocol {name} not found")
    return str(p)


DEFAULT_PROTOCOL = protocol_path()


class CFD2vec:
    def __init__(self, net: CFD2vecNet, device: str = "cpu"):
        self.net, self.device = net.to(device).eval(), device

    @classmethod
    def from_pretrained(cls, path: str, device: str = "auto") -> "CFD2vec":
        dev = pick_device(device)
        ck = torch.load(path, map_location="cpu", weights_only=False)      # moved to the device below
        net = CFD2vecNet(ModelConfig.from_checkpoint(ck["cfg"]), ck["stats"]["mean"], ck["stats"]["std"])
        net.load_state_dict(ck["model"])
        m = cls(net, dev); m.stats = ck["stats"]
        m.meta = {k: ck.get(k) for k in ("step", "objective", "provenance")}
        m.meta["input_schema"] = net.schema
        return m

    @classmethod
    def from_config(cls, model_config: str, stats_path: str, device: str = "auto") -> "CFD2vec":
        st = json.load(open(stats_path))
        m = cls(CFD2vecNet(ModelConfig.from_yaml(model_config), st["mean"], st["std"]), pick_device(device))
        m.stats = st
        return m

    def save(self, path: str, note: Optional[str] = None):
        """Save the weights with their lineage: step, objective and provenance of the checkpoint they were loaded
        from (`from_pretrained`), the input schema and an optional note. A fine-tuned model keeps its origin."""
        meta = dict(getattr(self, "meta", None) or {})
        torch.save(dict(model=self.net.state_dict(), cfg=self.net.cfg.to_dict(), stats=self.stats,
                        step=meta.get("step"), objective=meta.get("objective"), provenance=meta.get("provenance"),
                        input_schema=self.net.schema, note=note), path)

    def encode(self, case: Case, seed: int = 0, use_prior: bool = False) -> dict:
        from .tasks.predict import predict_case
        from .data.sampling import subsample
        tiny = subsample(case, [0])
        return predict_case(self.net, case, query=tiny, use_prior=use_prior, seed=seed, device=self.device,
                            return_embedding=True)

    def predict(self, case: Case, query: Optional[Case] = None, use_prior: bool = False, n_ensemble: int = 1,
                seed: int = 0) -> dict:
        from .tasks.predict import predict_case
        return predict_case(self.net, case, query=query, use_prior=use_prior, seed=seed, device=self.device,
                            n_ensemble=n_ensemble)

    def to_physical(self, fields, case: Case, **kw) -> dict:
        from .tasks.warmstart import SeedPolicy, to_physical
        return to_physical(fields, case, SeedPolicy(**kw))

    def finetune(self, case_paths: Sequence[str], protocol: str = DEFAULT_PROTOCOL, seed: int = 0,
                 use_prior: bool = False, log_path: Optional[str] = None, max_epochs: Optional[int] = None):
        from .train.finetune import finetune
        _, hist = finetune(self.net, case_paths, protocol, seed=seed, use_prior=use_prior, device=self.device,
                           log_path=log_path, max_epochs=max_epochs)
        self.net.eval()
        return hist


def input_keys(schema: int = 2) -> tuple:
    """Graph inputs of the exported model. Schema 1 reads target presence; schema 2 reads observed presence."""
    pres = ("presence",) if schema == 1 else ("ctx_obs",)
    return (("ctx_pos", "ctx_wd", "ctx_nrm", "ctx_h", "ctx_s", "ctx_cs", "ctx_field", "ctx_prior", "ctx_vis")
            + pres + ("prior_presence", "cs_flag", "cond_vec", "closure", "ground", "prior_kind",
                      "tok_centre", "tok_nb", "tok_scale", "tok_radius", "tok_reff",
                      "q_pos", "q_wd", "q_nrm", "q_h", "q_s", "q_cs", "q_prior"))


INPUT_KEYS = input_keys(2)


class _OnnxWrapper(torch.nn.Module):
    def __init__(self, net, keys):
        super().__init__(); self.net = net; self.keys = keys

    def forward(self, *xs):
        b = dict(zip(self.keys, xs))
        mem, pos, glob = self.net.encode(b)
        return self.net.decode(mem, pos, b), glob


def export_onnx(ckpt: str, out: str, opset: int = 17):
    """Export encoder + decoder as one ONNX graph. Inputs are a prepared sample (input_keys(schema), batch dimension
    first); token groups are computed outside the graph (cfd2vec.train.dataset.make_sample), so the graph is pure
    tensor algebra and runs in ONNX Runtime and in C++ solver bindings. Requires the `onnx` package.
    The graph has no token validity masks: it accepts cases with at least n_context points (every token and
    neighbour slot valid). Smaller cases need the PyTorch model."""
    import numpy as np
    from .tasks.predict import spec_from_config
    from .train.dataset import make_sample, to_torch
    from .schema import Case, Conditioning
    m = CFD2vec.from_pretrained(ckpt, device="cpu")
    cfg = m.net.cfg
    rng = np.random.default_rng(0)
    n = cfg.n_context
    pts = rng.uniform(-2, 2, (n, 3)).astype(np.float32); pts[:, 2] = np.abs(pts[:, 2])
    dummy = Case("export", "synthetic", pts, pts[:, 2].copy(), np.tile([0, 0, 1.0], (n, 1)).astype(np.float32),
                 None, np.zeros(4, bool), Conditioning(), 1.0, 1.0)
    s = to_torch(make_sample(dummy, spec_from_config(cfg, fixed_ratio=1.0, augment=False, use_prior=False), rng,
                             query_case=dummy))
    keys = input_keys(m.net.schema)
    xs = tuple(s[k][None] for k in keys)
    dyn = {k: {0: "batch"} for k in keys}
    for k in keys:
        if k.startswith("q_"):
            dyn[k][1] = "n_query"
    torch.onnx.export(_OnnxWrapper(m.net, keys).eval(), xs, out, input_names=list(keys),
                      output_names=["fields", "embedding"], dynamic_axes=dyn, opset_version=opset, dynamo=False)
    return out
