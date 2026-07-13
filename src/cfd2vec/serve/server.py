"""gRPC inference server: keeps one model resident on the accelerator and turns uploaded solver cases into initial
fields.

    cfd2vec serve --ckpt runs/<run>/last.pt --device mps --host 127.0.0.1 --port 50551

One request is served at a time (one model, one accelerator), so timings are not mixed between requests.
Recommended: bind to 127.0.0.1 and reach it from a local VM through the host-forwarding name of the VM runtime;
the service has no authentication.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from concurrent import futures

import numpy as np
import torch

from . import wire

log = logging.getLogger("cfd2vec.serve")


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Predictor:
    """Model + solver adapter. `predict_frame` maps a request frame to a response frame."""

    def __init__(self, ckpt: str, device: str = "auto", workdir: str | None = None):
        from ..api import CFD2vec
        from ..solvers.openfoam import OpenFOAMAdapter
        self.m = CFD2vec.from_pretrained(ckpt, device=device)
        self.device = self.m.device
        self.ckpt = ckpt
        self.ad = OpenFOAMAdapter()
        self.workdir = workdir
        self.lock = threading.Lock()
        self.warm = False
        self.n_served = 0
        self.info = dict(ckpt=os.path.abspath(ckpt), device=self.device, step=self.m.meta.get("step"),
                         objective=self.m.meta.get("objective"),
                         params=int(sum(p.numel() for p in self.m.net.parameters())),
                         torch=torch.__version__)

    # ---- inference with per-stage timing (same computation as tasks.predict.predict_case, n_ensemble 1) ----
    @torch.no_grad()
    def _predict(self, case, use_prior: bool, seed: int, t: dict) -> np.ndarray:
        from ..data.sampling import strip_targets
        from ..schema import decode_fields
        from ..tasks.predict import spec_from_config
        from ..train.dataset import make_sample, to_torch
        net = self.m.net
        net.eval()
        s0 = time.perf_counter()
        spec = spec_from_config(net.cfg, fixed_ratio=1.0, augment=False, use_prior=use_prior)
        s = to_torch(make_sample(strip_targets(case), spec, np.random.default_rng(seed), query_case=case))
        t["sample"] = time.perf_counter() - s0
        s0 = time.perf_counter()
        b = {k: v[None].to(self.device) for k, v in s.items()}
        mem, pos, glob = net.encode(b)
        out = net.decode(mem, pos, b, chunk=16384)
        P = decode_fields(net.unstandardise(out))[0].float().cpu().numpy()
        _sync(self.device)
        t["forward"] = time.perf_counter() - s0
        if not np.isfinite(P).all():
            raise FloatingPointError(f"{case.case_id}: non-finite predicted fields")
        return P

    def warmup(self, n: int | None = None):
        """One synthetic forward pass so the first real request does not pay kernel compilation / allocator growth."""
        from ..schema import Case, Conditioning
        cfg = self.m.net.cfg
        n = n or cfg.n_context + 4096
        rng = np.random.default_rng(0)
        pts = rng.uniform(-2, 2, (n, 3)).astype(np.float32); pts[:, 2] = np.abs(pts[:, 2])
        c = Case("warmup", "synthetic", pts, pts[:, 2].copy(), np.tile([0, 0, 1.0], (n, 1)).astype(np.float32),
                 None, np.zeros(4, bool), Conditioning(), 1.0, 1.0)
        t = {}
        s0 = time.perf_counter()
        self._predict(c, False, 0, t)
        self.warm = True
        self.info["warmup_s"] = round(time.perf_counter() - s0, 3)
        return self.info["warmup_s"]

    def predict_frame(self, buf: bytes, t_receive: float) -> bytes:
        from ..schema import Conditioning
        from ..tasks.warmstart import SeedPolicy, attach_prior, to_physical
        t = dict(receive=t_receive)
        t_all = time.perf_counter()
        tmp = tempfile.mkdtemp(prefix="cfd2vec_serve_", dir=self.workdir)
        try:
            h, payload = wire.unframe(buf)
            s0 = time.perf_counter()
            wire.untar(payload, tmp)
            cdir = os.path.join(tmp, "case")
            open(os.path.join(cdir, "case.foam"), "w").close()
            pdir = os.path.join(tmp, "prior")
            if os.path.isdir(pdir):
                open(os.path.join(pdir, "prior.foam"), "w").close()
            t["unpack"] = time.perf_counter() - s0
            want = list(h.get("fields") or ["U"])
            time_dir = str(h.get("time", "0"))
            cond = Conditioning(**(h.get("cond") or {}))
            s0 = time.perf_counter()
            case = self.ad.read_case(cdir, U_ref=float(h["U_ref"]), L_ref=h.get("L_ref"), cond=copy.deepcopy(cond),
                                     case_id=h.get("case_id"))
            if h.get("nu"):
                case.cond.log10_re = float(math.log10(case.U_ref * case.L_ref / float(h["nu"])))
            use_prior = bool(h.get("use_prior"))
            if use_prior:
                if not os.path.isdir(pdir):
                    raise ValueError("use_prior requested but the payload has no prior/ case")
                prior = self.ad.read_case(pdir, U_ref=case.U_ref, L_ref=case.L_ref, cond=copy.deepcopy(cond),
                                          with_fields=True, origin=np.asarray(case.meta["origin"]))
                wp = copy.copy(case); wp.cond = copy.deepcopy(case.cond)
                case = attach_prior(wp, prior)
            t["read_case"] = time.perf_counter() - s0
            P = self._predict(case, use_prior, int(h.get("seed", 0)), t)
            s0 = time.perf_counter()
            phys = to_physical(P, case, SeedPolicy())
            missing = [f for f in want if f not in phys]
            if missing:
                raise KeyError(f"fields not produced by the model / seed policy: {missing}")
            t["to_physical"] = time.perf_counter() - s0
            s0 = time.perf_counter()
            self.ad.write_initial(cdir, {f: phys[f] for f in want}, time=time_dir, backup=False)
            t["write"] = time.perf_counter() - s0
            s0 = time.perf_counter()
            body = wire.tar_paths([(os.path.join(cdir, time_dir, f), f"{time_dir}/{f}") for f in want])
            t["pack"] = time.perf_counter() - s0
            t["total"] = time.perf_counter() - t_all + t_receive
            out = dict(ok=True, case_id=case.case_id, n_cells=int(case.n), L_ref=float(case.L_ref),
                       log10_re=case.cond.log10_re, use_prior=use_prior, fields=want, device=self.device,
                       timings_s={k: round(v, 4) for k, v in t.items()},
                       model={k: self.info[k] for k in ("step", "objective", "params")})
            return wire.frame(out, body)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _service(pred: Predictor):
    import grpc

    def predict(request_iterator, context):
        s0 = time.perf_counter()
        buf = wire.join(wire.decode_chunk(m) for m in request_iterator)
        t_recv = time.perf_counter() - s0
        with pred.lock:
            try:
                resp = pred.predict_frame(buf, t_recv)
                pred.n_served += 1
                h, _ = wire.unframe(resp)
                log.info("served %s: %d cells, %s", h["case_id"], h["n_cells"], json.dumps(h["timings_s"]))
            except Exception as e:                         # reported to the client, server keeps running
                log.exception("request failed")
                resp = wire.frame(dict(ok=False, error=f"{type(e).__name__}: {e}"))
        for part in wire.split(resp):
            yield wire.encode_chunk(part)

    def health(request, context):
        return wire.encode_chunk(wire.frame(dict(pred.info, warm=pred.warm, served=pred.n_served)))

    ident = lambda b: b                                     # noqa: E731  (Chunk bytes are coded in wire)
    return grpc.method_handlers_generic_handler(wire.SERVICE, {
        "Predict": grpc.stream_stream_rpc_method_handler(predict, request_deserializer=ident,
                                                         response_serializer=ident),
        "Health": grpc.unary_unary_rpc_method_handler(health, request_deserializer=ident,
                                                      response_serializer=ident),
    })


def serve(ckpt: str, device: str = "auto", host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
          warmup: bool = True, workdir: str | None = None, threads: int | None = None):
    import grpc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if threads:
        torch.set_num_threads(threads)
    pred = Predictor(ckpt, device=device, workdir=workdir)
    log.info("model %s: %d params, step %s, device %s", pred.info["ckpt"], pred.info["params"], pred.info["step"],
             pred.device)
    if warmup:
        log.info("warm-up forward pass: %.2f s", pred.warmup())
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2),
                      options=[("grpc.max_receive_message_length", 8 * wire.CHUNK_BYTES),
                               ("grpc.max_send_message_length", 8 * wire.CHUNK_BYTES)])
    srv.add_generic_rpc_handlers((_service(pred),))
    addr = host if host.startswith("unix:") else f"{host}:{port}"      # unix:/path serves on a local socket
    if srv.add_insecure_port(addr) == 0:
        raise OSError(f"could not bind {addr}")
    srv.start()
    log.info("listening on %s (Ctrl-C to stop)", addr)
    try:
        srv.wait_for_termination()
    except KeyboardInterrupt:
        srv.stop(grace=2)
