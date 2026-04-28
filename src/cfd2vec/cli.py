"""Command-line interface:  cfd2vec pretrain | device-check | finetune | evaluate | warmstart | export-onnx | serve"""
from __future__ import annotations

import argparse
import sys

from .paths import expand


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cfd2vec")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain", help="masked field modelling (or the supervised control)")
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--objective", choices=["masked", "supervised"], default="masked")
    p.add_argument("--device", default="auto"); p.add_argument("--threads", type=int, default=0)
    p.add_argument("--max-minutes", type=float); p.add_argument("--max-steps", type=int); p.add_argument("--init")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt (model, optimiser, step)")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--num-workers", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-checkpoint", choices=["on", "off"],
                   help="override the model config: off keeps activations (faster, more memory); weights unaffected")

    d = sub.add_parser("device-check", help="accelerator availability, CPU parity, throughput and memory per batch size")
    d.add_argument("--model-config", default="configs/model_S_urban.yaml")
    d.add_argument("--stats", default="configs/channel_stats.json")
    d.add_argument("--manifest", default="${CFD2VEC_DATA}/cfd2vec_shards/manifest_aero_urban.parquet")
    d.add_argument("--device", default="auto"); d.add_argument("--batch-sizes", default="1,2,4,8")
    d.add_argument("--memory-fraction", type=float, default=0.9)
    d.add_argument("--max-step-s", type=float, default=60.0)
    d.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    d.add_argument("--grad-checkpoint", choices=["on", "off"])
    d.add_argument("--out", default="runs/device_check.json")

    f = sub.add_parser("finetune", help="few-shot fine-tuning under the frozen protocol")
    f.add_argument("--protocol", help="protocol YAML (default: the protocol shipped with the package)")
    f.add_argument("--init", help="pretrained checkpoint; omit for from-scratch")
    f.add_argument("--model-config", default="configs/model_pilot.yaml")
    f.add_argument("--stats", default="configs/channel_stats.json")
    f.add_argument("--cases", nargs="+", required=True, help="Case shard paths (the N fine-tuning cases)")
    f.add_argument("--seed", type=int, default=0); f.add_argument("--out", required=True)
    f.add_argument("--use-prior", action="store_true"); f.add_argument("--device", default="auto")

    e = sub.add_parser("evaluate", help="full-field metrics on held-out cases")
    e.add_argument("--ckpt", required=True); e.add_argument("--cases", nargs="+", required=True)
    e.add_argument("--out", required=True); e.add_argument("--use-prior", action="store_true")
    e.add_argument("--device", default="auto")

    w = sub.add_parser("warmstart", help="write solver initial fields for a case directory")
    w.add_argument("--ckpt", required=True); w.add_argument("--case-dir", required=True)
    w.add_argument("--solver", default="openfoam"); w.add_argument("--fields", default="U")
    w.add_argument("--U-ref", type=float, required=True); w.add_argument("--L-ref", type=float, required=True)
    w.add_argument("--inflow-dir", default="1,0,0"); w.add_argument("--closure", default="k-epsilon")
    w.add_argument("--alpha", type=float); w.add_argument("--turb-intensity", type=float)
    w.add_argument("--prior-dir", help="case directory of a low-fidelity solve to use as prior")

    x = sub.add_parser("export-onnx"); x.add_argument("--ckpt", required=True); x.add_argument("--out", required=True)

    v = sub.add_parser("serve", help="gRPC inference server for solver-side clients (proto/cfd2vec_serve.proto)")
    v.add_argument("--ckpt", required=True); v.add_argument("--device", default="auto")
    v.add_argument("--host", default="127.0.0.1"); v.add_argument("--port", type=int, default=50551)
    v.add_argument("--no-warmup", action="store_true"); v.add_argument("--threads", type=int)
    v.add_argument("--workdir", help="scratch directory for uploaded cases (default: system temp)")

    a = ap.parse_args(argv)
    if a.cmd == "pretrain":
        from .train.pretrain import pretrain
        pretrain(a.config, a.out, a.objective, a.device, a.threads, a.max_minutes, a.max_steps, a.init,
                 resume=a.resume, precision=a.precision, num_workers=a.num_workers, batch_size=a.batch_size,
                 grad_checkpoint=None if a.grad_checkpoint is None else a.grad_checkpoint == "on")
    elif a.cmd == "device-check":
        import json
        from .train.device_check import device_check
        device_check(a.model_config, a.stats, expand(a.manifest), a.device, tuple(int(x) for x in a.batch_sizes.split(",")),
                     out=a.out, memory_fraction=a.memory_fraction, max_step_s=a.max_step_s, precision=a.precision,
                     grad_checkpoint=None if a.grad_checkpoint is None else a.grad_checkpoint == "on")
    elif a.cmd == "finetune":
        from .train.finetune import finetune_cli
        finetune_cli(a)
    elif a.cmd == "evaluate":
        from .eval.evaluate import evaluate_cli
        evaluate_cli(a)
    elif a.cmd == "warmstart":
        from .tasks.warmstart import warmstart_cli
        warmstart_cli(a)
    elif a.cmd == "export-onnx":
        from .api import export_onnx
        export_onnx(a.ckpt, a.out)
    elif a.cmd == "serve":
        from .serve.server import serve
        serve(a.ckpt, device=a.device, host=a.host, port=a.port, warmup=not a.no_warmup, workdir=a.workdir,
              threads=a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
