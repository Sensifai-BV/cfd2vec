"""Minimal reader of an OpenFOAM polyMesh (points, faces, owner, neighbour, boundary), ascii or binary.

Its one job is to establish face-owner connectivity for wall-normal orientation from the mesh files themselves,
so the adjacent fluid cell of a boundary face is known by identity, not inferred from distances. Reader faces
(from VTK) are matched to polyMesh faces of the same patch by face-centre identity, one to one, within a small
fraction of the face width; anything that does not verify raises, and the caller keeps the reader's orientation
and marks it unverified. Nothing is written.
"""
from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np

_HEADER = re.compile(rb"FoamFile\s*\{(.*?)\}", re.S)
_ENTRY = re.compile(rb'(\w+)\s+("(?:[^"\\]|\\.)*"|[^;]+);')     # a quoted value may hold semicolons
_INT = re.compile(rb"\d+")
_PATCH = re.compile(rb"([\w.\-:]+)\s*\{([^}]*)\}", re.S)


def _header(data: bytes):
    m = _HEADER.search(data[:8192])
    if not m:
        raise ValueError("no FoamFile header")
    h = {k.decode(): v.decode().strip().strip('"') for k, v in _ENTRY.findall(m.group(1))}
    return h, m.end()


def _dtypes(h: dict):
    """Label and scalar dtypes from the header's `arch "LSB;label=32;scalar=64"` entry (OpenFOAM's defaults when
    the entry is absent). An unrecognised declaration raises rather than guessing a width."""
    arch = h.get("arch", "LSB;label=32;scalar=64")
    parts = [x.strip() for x in arch.split(";") if x.strip()]
    order_word = parts[0] if parts else "LSB"
    kv = dict(x.split("=", 1) for x in parts[1:] if "=" in x)
    label, scalar = kv.get("label", "32"), kv.get("scalar", "64")
    if order_word not in ("LSB", "MSB") or label not in ("32", "64") or scalar not in ("32", "64"):
        raise ValueError(f"unsupported arch declaration {arch!r}")
    order = "<" if order_word == "LSB" else ">"
    lb, sc = int(label) // 8, int(scalar) // 8
    return f"{order}i{lb}", f"{order}f{sc}", lb, sc


class _Cursor:
    def __init__(self, data: bytes, pos: int):
        self.d, self.p = data, pos

    def skip(self):
        d = self.d
        while self.p < len(d):
            if d[self.p] in b" \t\r\n":
                self.p += 1
            elif d.startswith(b"//", self.p):
                e = d.find(b"\n", self.p); self.p = len(d) if e < 0 else e + 1
            elif d.startswith(b"/*", self.p):
                e = d.find(b"*/", self.p); self.p = len(d) if e < 0 else e + 2
            else:
                break

    def int(self) -> int:
        self.skip()
        m = _INT.match(self.d, self.p)
        if not m:
            raise ValueError(f"expected a count at byte {self.p}")
        self.p = m.end()
        return int(m.group())

    def expect(self, ch: bytes):
        self.skip()
        if self.d[self.p:self.p + 1] != ch:
            raise ValueError(f"expected {ch!r} at byte {self.p}")
        self.p += 1

    def raw(self, n: int) -> bytes:
        b = self.d[self.p:self.p + n]
        if len(b) != n:
            raise ValueError("truncated binary block")
        self.p += n
        return b

    def ascii_list(self) -> bytes:
        """Text between the '(' just consumed and its matching ')'."""
        depth, start, d = 1, self.p, self.d
        while depth:
            o, c = d.find(b"(", self.p), d.find(b")", self.p)
            if c < 0:
                raise ValueError("unterminated list")
            if 0 <= o < c:
                depth += 1; self.p = o + 1
            else:
                depth -= 1; self.p = c + 1
        return d[start:self.p - 1]


def _ascii_numbers(txt: bytes, dtype) -> np.ndarray:
    return np.array(txt.replace(b"(", b" ").replace(b")", b" ").split(), dtype=dtype)


def read_points(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    h, pos = _header(data)
    _, fdt, _, sc = _dtypes(h)
    cur = _Cursor(data, pos)
    n = cur.int(); cur.expect(b"(")
    if h.get("format") == "binary":
        pts = np.frombuffer(cur.raw(n * 3 * sc), dtype=fdt).astype(np.float64).reshape(n, 3)
        cur.expect(b")")
    else:
        pts = _ascii_numbers(cur.ascii_list(), np.float64).reshape(n, 3)
    return pts


def read_labels(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    h, pos = _header(data)
    ldt, _, lb, _ = _dtypes(h)
    cur = _Cursor(data, pos)
    n = cur.int(); cur.expect(b"(")
    if h.get("format") == "binary":
        a = np.frombuffer(cur.raw(n * lb), dtype=ldt).astype(np.int64)
        cur.expect(b")")
    else:
        a = _ascii_numbers(cur.ascii_list(), np.int64)
    if a.size != n:
        raise ValueError(f"{path}: {a.size} labels for a declared {n}")
    return a


def read_faces(path: str):
    """(offsets (F+1,), flat point ids) of the face list, from a faceList (ascii `4(a b c d)`) or a
    faceCompactList (offsets then flat labels, the binary layout)."""
    data = open(path, "rb").read()
    h, pos = _header(data)
    ldt, _, lb, _ = _dtypes(h)
    cur = _Cursor(data, pos)
    binary = h.get("format") == "binary"
    if "Compact" in h.get("class", "") or binary:
        n = cur.int(); cur.expect(b"(")
        off = np.frombuffer(cur.raw(n * lb), dtype=ldt).astype(np.int64) if binary else \
            _ascii_numbers(cur.ascii_list(), np.int64)
        if binary:
            cur.expect(b")")
        m = cur.int(); cur.expect(b"(")
        flat = np.frombuffer(cur.raw(m * lb), dtype=ldt).astype(np.int64) if binary else \
            _ascii_numbers(cur.ascii_list(), np.int64)
        if binary:
            cur.expect(b")")
        if off.size != n or flat.size != m or off[0] != 0 or off[-1] != m:
            raise ValueError(f"{path}: inconsistent compact face list")
        return off, flat
    n = cur.int(); cur.expect(b"(")
    tok = _ascii_numbers(cur.ascii_list(), np.int64)
    off = np.zeros(n + 1, np.int64)
    p = 0
    for i in range(n):                            # `k(id ...)`: a count token then k ids
        k = int(tok[p]); off[i + 1] = off[i] + k; p += 1 + k
    if p != tok.size:
        raise ValueError(f"{path}: face list does not parse to {n} faces")
    keep = np.ones(tok.size, bool); keep[off[:-1] + np.arange(n)] = False
    return off, tok[keep]


def read_boundary(path: str) -> dict:
    """patch name -> dict(type, nFaces, startFace) in file order."""
    data = open(path, "rb").read()
    _, pos = _header(data)
    body = re.sub(rb"//[^\n]*", b"", data[pos:])
    out = {}
    for name, blk in _PATCH.findall(body):
        e = {k.decode(): v.decode().strip() for k, v in _ENTRY.findall(blk)}
        if "startFace" in e and "nFaces" in e:
            out[name.decode()] = dict(type=e.get("type", "patch"), nFaces=int(e["nFaces"]),
                                      startFace=int(e["startFace"]))
    if not out:
        raise ValueError(f"{path}: no patches found")
    return out


def read_polymesh(mesh_dir: str) -> dict:
    pts = read_points(os.path.join(mesh_dir, "points"))
    off, flat = read_faces(os.path.join(mesh_dir, "faces"))
    owner = read_labels(os.path.join(mesh_dir, "owner"))
    neigh = read_labels(os.path.join(mesh_dir, "neighbour"))
    patches = read_boundary(os.path.join(mesh_dir, "boundary"))
    n_faces = off.size - 1
    if owner.size != n_faces or neigh.size > n_faces or flat.max() >= len(pts):
        raise ValueError(f"{mesh_dir}: points / faces / owner / neighbour are inconsistent")
    return dict(points=pts, face_offsets=off, face_points=flat, owner=owner, neighbour=neigh, patches=patches)


def face_centres(mesh: dict, faces: np.ndarray | None = None) -> np.ndarray:
    """Vertex mean of each face (all faces, or the given face ids)."""
    off, flat, pts = mesh["face_offsets"], mesh["face_points"], mesh["points"]
    if faces is None:
        faces = np.arange(off.size - 1)
    faces = np.asarray(faces, np.int64)
    counts = off[faces + 1] - off[faces]
    idx = np.concatenate([np.arange(off[f], off[f + 1]) for f in faces]) if faces.size else np.zeros(0, np.int64)
    sums = np.add.reduceat(pts[flat[idx]], np.concatenate([[0], np.cumsum(counts)[:-1]]), axis=0) if faces.size \
        else np.zeros((0, 3))
    return sums / counts[:, None]


def cell_centres(mesh: dict) -> np.ndarray:
    """Topological cell centre: mean of the centres of the faces bounding each cell (owner and neighbour faces)."""
    fc = face_centres(mesh)
    owner, neigh = mesh["owner"], mesh["neighbour"]
    n_cells = int(max(owner.max(), neigh.max() if neigh.size else -1)) + 1
    acc = np.zeros((n_cells, 3)); cnt = np.zeros(n_cells)
    np.add.at(acc, owner, fc); np.add.at(cnt, owner, 1.0)
    np.add.at(acc, neigh, fc[:neigh.size]); np.add.at(cnt, neigh, 1.0)
    if (cnt == 0).any():
        raise ValueError("a cell has no faces")
    return acc / cnt[:, None]


def verified_owner_centres(case_dir: str, patches: Sequence[str], reader_face_centres: Sequence[np.ndarray],
                           reader_face_widths: Sequence[np.ndarray], rel_tol: float = 1e-2):
    """Owner (adjacent fluid) cell centre of every reader face of the given wall patches, established from the
    polyMesh: each reader face is matched to a polyMesh face of the same patch by centre identity (nearest
    polyMesh face centre within `rel_tol` of the face width, one to one), and the owner's topological centre is
    returned. Raises ValueError naming the first patch that does not verify (count mismatch, an unmatched face, a
    face matched twice, as for the two coincident faces of a baffle). Returns (owner centres, report)."""
    from scipy.spatial import cKDTree
    mesh = read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))
    cc = cell_centres(mesh)
    out, worst = [], 0.0
    for name, rf, rw in zip(patches, reader_face_centres, reader_face_widths):
        if name not in mesh["patches"]:
            raise ValueError(f"{name}: not a patch of the polyMesh")
        p = mesh["patches"][name]
        ids = np.arange(p["startFace"], p["startFace"] + p["nFaces"])
        rf = np.asarray(rf, np.float64).reshape(-1, 3)
        if len(rf) != ids.size:
            raise ValueError(f"{name}: {len(rf)} reader faces for {ids.size} polyMesh faces")
        if ids.size == 0:
            out.append(np.zeros((0, 3))); continue
        pc = face_centres(mesh, ids)
        d, j = cKDTree(pc).query(rf, workers=-1)
        tol = float(rel_tol) * np.asarray(rw, np.float64).ravel()
        if (d > tol).any():
            k = int(np.argmax(d - tol))
            raise ValueError(f"{name}: reader face {k} is {d[k]:.3g} from the nearest polyMesh face centre "
                             f"(tolerance {tol[k]:.3g}); face order or geometry differs")
        if np.unique(j).size != ids.size:
            raise ValueError(f"{name}: two reader faces match one polyMesh face (coincident faces?)")
        worst = max(worst, float((d / np.maximum(tol, 1e-300)).max()))
        out.append(cc[mesh["owner"][ids[j]]])
    return np.concatenate(out) if out else np.zeros((0, 3)), \
        dict(verified=True, matched_faces=int(sum(len(o) for o in out)), rel_tol=float(rel_tol),
             worst_match_over_tol=worst, polymesh_format="binary" if _is_binary(case_dir) else "ascii")


def _is_binary(case_dir: str) -> bool:
    h, _ = _header(open(os.path.join(case_dir, "constant", "polyMesh", "faces"), "rb").read())
    return h.get("format") == "binary"
