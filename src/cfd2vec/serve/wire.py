"""Wire format of the CFD2vec inference service (proto/cfd2vec_serve.proto). Standard library only, so a solver-side
client needs nothing beyond grpcio.

Request header (JSON):
    case_id      str
    U_ref        float   reference speed [m/s]
    nu           float   kinematic viscosity [m^2/s]; log10 Re = log10(U_ref L_ref / nu)
    L_ref        float | null   reference length [m]; null = mean building / body height from the mesh
    cond         dict    Conditioning fields (closure, abl_alpha, turb_intensity, ground, rotation_ok, ...)
    fields       list    solver fields to return, subset of U, p, k, epsilon, omega, nut
    use_prior    bool    condition on the low-fidelity solution under prior/ (payload must contain it)
    time         str     initial-field directory to write (default "0")
    seed         int     context-sampling seed (default 0)
Response header (JSON):
    ok, error, case_id, n_cells, L_ref, log10_re, fields, timings_s {receive, unpack, read_case, sample, forward,
    to_physical, write, pack, total}, device, model {step, objective, params}
"""
from __future__ import annotations

import io
import json
import os
import struct
import tarfile
from typing import Iterable, Iterator, Tuple

SERVICE = "cfd2vec.serve.v1.Predictor"
PREDICT = f"/{SERVICE}/Predict"
HEALTH = f"/{SERVICE}/Health"
CHUNK_BYTES = 1 << 20                    # 1 MiB per message, well below the gRPC 4 MiB default
DEFAULT_PORT = 50551


# ---- protobuf encoding of `message Chunk { bytes data = 1; }` ----
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_chunk(data: bytes) -> bytes:
    return b"\x0a" + _varint(len(data)) + data if data else b""


def decode_chunk(msg: bytes) -> bytes:
    if not msg:
        return b""
    if msg[0] != 0x0A:
        raise ValueError(f"unexpected protobuf tag {msg[0]:#x}")
    n, shift, i = 0, 0, 1
    while True:
        b = msg[i]; i += 1
        n |= (b & 0x7F) << shift; shift += 7
        if not b & 0x80:
            break
    if i + n != len(msg):
        raise ValueError("truncated Chunk message")
    return msg[i:i + n]


# ---- frames ----
def frame(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + payload


def unframe(buf: bytes) -> Tuple[dict, bytes]:
    (n,) = struct.unpack("<Q", buf[:8])
    return json.loads(buf[8:8 + n].decode()), buf[8 + n:]


def split(buf: bytes, size: int = CHUNK_BYTES) -> Iterator[bytes]:
    """Raw frame bytes -> payloads of successive Chunk messages."""
    mv = memoryview(buf)
    for i in range(0, max(len(buf), 1), size):
        yield bytes(mv[i:i + size])


def join(chunks: Iterable[bytes]) -> bytes:
    return b"".join(chunks)


# ---- tar helpers ----
def tar_paths(entries: Iterable[Tuple[str, str]]) -> bytes:
    """entries: (path on disk, name inside the archive). Directories are added recursively."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as t:
        for src, arc in entries:
            t.add(src, arcname=arc)
    return bio.getvalue()


def untar(payload: bytes, dest: str) -> list:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r") as t:
        names = t.getnames()
        for n in names:                  # archives come from the peer; refuse paths outside dest
            full = os.path.realpath(os.path.join(dest, n))
            if not full.startswith(os.path.realpath(dest) + os.sep) and full != os.path.realpath(dest):
                raise ValueError(f"archive member escapes the destination: {n}")
        t.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dest)
    return names


def case_entries(case_dir: str, time: str = "0", arc_root: str = "case",
                 skip_fields: Tuple[str, ...] = ("C", "Ccx", "Ccy", "Ccz", "Vc", "phi", "Phi")) -> list:
    """Archive entries of an OpenFOAM case: constant/, system/ and the templates in <time>/ (derived geometry
    fields written by postProcess utilities are skipped; the server recomputes geometry from the mesh)."""
    ent = [(os.path.join(case_dir, "constant"), f"{arc_root}/constant"),
           (os.path.join(case_dir, "system"), f"{arc_root}/system")]
    tdir = os.path.join(case_dir, time)
    for f in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, f)
        if os.path.isfile(p) and f not in skip_fields:
            ent.append((p, f"{arc_root}/{time}/{f}"))
    return ent


def latest_time(case_dir: str) -> str:
    ts = []
    for d in os.listdir(case_dir):
        try:
            ts.append((float(d), d))
        except ValueError:
            pass
    if not ts:
        raise FileNotFoundError(f"{case_dir}: no time directories")
    return max(ts)[1]
