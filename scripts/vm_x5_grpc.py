"""Warm-start benchmark with inference served over gRPC (solver side: python3 + grpcio + OpenFOAM).

    python3 vm_x5_grpc.py --src <bench>/<case> --ref <x5_root>/<case> --out <x5_gpu_root>/<case> \
        --server host.orb.internal:50551 [--arms cfd2vec_U,...] [--latency-repeats 3] [--solve]

Per model arm, the client uploads the pristine case (mesh, system, 0/ templates; plus the coarse solve for prior
arms), receives the predicted initial fields and writes them into a fresh arm directory. The full client wall-clock
from packing to written files is charged to the arm (the coarse solve time is added for prior arms, as in the CPU
protocol). protocol.json and probes.json are copied from --ref so monitors and criterion are identical.
With --solve the arms are then run by vm_x5_bench.py under the same fixed budget.
Timings: <out>/grpc_timings.json (client stages + server stages per request, and repeated-latency samples).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="benchmark case with fine/ (and coarse/ for prior arms), params.json")
ap.add_argument("--ref", required=True, help="prepared CPU benchmark root of the same case (protocol, probes)")
ap.add_argument("--out", required=True)
ap.add_argument("--server", default="host.orb.internal:50551")
ap.add_argument("--arms", default="cfd2vec_U,cfd2vec_Up,cfd2vec_Upkeps,supp_cfd2vec_prior_U")
ap.add_argument("--latency-repeats", type=int, default=0, help="extra timed geometry-only requests, not written")
ap.add_argument("--solve", action="store_true")
ap.add_argument("--pydeps", default="", help="directory holding an extracted grpcio wheel, prepended to sys.path")
ap.add_argument("--env", default="source /opt/openfoam14/etc/bashrc")
a = ap.parse_args()
if a.pydeps:
    sys.path.insert(0, a.pydeps)
import grpc  # noqa: E402
from cfd2vec.serve import wire  # noqa: E402

ARMS = {
    "cfd2vec_U": dict(fields=["U"], use_prior=False),
    "cfd2vec_Up": dict(fields=["U", "p"], use_prior=False),
    "cfd2vec_Upkeps": dict(fields=["U", "p", "k", "epsilon"], use_prior=False),
    "supp_cfd2vec_prior_U": dict(fields=["U"], use_prior=True),
}
arms = [x for x in a.arms.split(",") if x]
unknown = [x for x in arms if x not in ARMS]
if unknown:
    raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
prm = json.load(open(os.path.join(a.src, "params.json")))["params"]
meta = json.load(open(os.path.join(a.src, "meta.json")))
fine, coarse = os.path.join(a.src, "fine"), os.path.join(a.src, "coarse")
cond = dict(closure="k-epsilon", abl_alpha=prm["alpha"], turb_intensity=prm.get("I"), ground="stationary",
            rotation_ok=True)
opts = [("grpc.max_receive_message_length", 8 * wire.CHUNK_BYTES),
        ("grpc.max_send_message_length", 8 * wire.CHUNK_BYTES)]
ch = grpc.insecure_channel(a.server, options=opts)
grpc.channel_ready_future(ch).result(timeout=30)
ident = lambda b: b  # noqa: E731
health = ch.unary_unary(wire.HEALTH, request_serializer=ident, response_deserializer=ident)
predict = ch.stream_stream(wire.PREDICT, request_serializer=ident, response_deserializer=ident)
info, _ = wire.unframe(wire.decode_chunk(health(wire.encode_chunk(wire.frame({})), timeout=30)))
print("server:", json.dumps(info), flush=True)


def request(spec: dict, dest_dir: str | None):
    """One end-to-end prediction. Returns (client timings, server header)."""
    t = {}
    t0 = s0 = time.perf_counter()
    ent = wire.case_entries(fine, "0", "case")
    if spec["use_prior"]:
        lt = wire.latest_time(coarse)
        ent += [(os.path.join(coarse, "constant"), "prior/constant"), (os.path.join(coarse, "system"), "prior/system"),
                (os.path.join(coarse, lt), f"prior/{lt}")]
    hdr = dict(case_id=os.path.basename(os.path.normpath(a.src)), U_ref=prm["Uref"], nu=prm["nu"], L_ref=None,
               cond=cond, fields=spec["fields"], use_prior=spec["use_prior"], time="0", seed=0)
    buf = wire.frame(hdr, wire.tar_paths(ent))
    t["pack"] = time.perf_counter() - s0
    t["upload_MB"] = len(buf) / 2 ** 20
    s0 = time.perf_counter()
    resp = wire.join(wire.decode_chunk(m) for m in
                     predict((wire.encode_chunk(p) for p in wire.split(buf)), timeout=1800))
    t["rpc"] = time.perf_counter() - s0
    h, body = wire.unframe(resp)
    if not h.get("ok"):
        raise RuntimeError(f"server error: {h.get('error')}")
    t["download_MB"] = len(resp) / 2 ** 20
    s0 = time.perf_counter()
    if dest_dir is not None:
        wire.untar(body, dest_dir)
    t["unpack_write"] = time.perf_counter() - s0
    t["end_to_end"] = time.perf_counter() - t0
    t["transfer_and_queue"] = t["rpc"] - h["timings_s"]["total"] + h["timings_s"]["receive"]
    return {k: round(v, 4) for k, v in t.items()}, h


os.makedirs(a.out, exist_ok=True)
for f in ("protocol.json", "probes.json"):
    shutil.copy(os.path.join(a.ref, f), os.path.join(a.out, f))
log = dict(server=info, src=a.src, arms={}, latency=[])
for _ in range(a.latency_repeats):                       # repeated geometry-only requests, discarded fields
    ct, h = request(ARMS["cfd2vec_U"], None)
    log["latency"].append(dict(client=ct, server=h["timings_s"]))
    print("latency", json.dumps(ct), json.dumps(h["timings_s"]), flush=True)
for name in arms:
    d = os.path.join(a.out, name)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for sub in ("constant", "system", "0"):                   # same arm layout as the CPU preparation
        shutil.copytree(os.path.join(fine, sub), os.path.join(d, sub))
    for f in ("C", "Ccx", "Ccy", "Ccz", "Vc"):
        if os.path.exists(os.path.join(d, "0", f)):
            os.remove(os.path.join(d, "0", f))
    ct, h = request(ARMS[name], d)
    extra = float(meta.get("exec_coarse_s", 0.0)) if ARMS[name]["use_prior"] else 0.0
    charge = ct["end_to_end"] + extra
    json.dump(dict(arm=name, pre="", charge_s=charge, inference="grpc", server_device=h["device"],
                   client_s=ct, server_s=h["timings_s"], coarse_solve_s=extra), open(os.path.join(d, "arm.json"), "w"),
              indent=1)
    log["arms"][name] = dict(client=ct, server=h["timings_s"], charge_s=round(charge, 3), n_cells=h["n_cells"],
                             L_ref=h["L_ref"], log10_re=h["log10_re"], fields=h["fields"])
    print(name, json.dumps(log["arms"][name]), flush=True)
json.dump(log, open(os.path.join(a.out, "grpc_timings.json"), "w"), indent=1)
if a.solve:
    r = subprocess.run([sys.executable, os.path.join(HERE, "vm_x5_bench.py"), a.out, "--arms", ",".join(arms),
                        "--env", a.env])
    sys.exit(r.returncode)
