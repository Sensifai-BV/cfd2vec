"""A small OpenFOAM polyMesh writer for tests: a Cartesian grid with arbitrary grid lines (so cells can be stretched)
and solid cells removed. Writes points, faces, owner, neighbour, boundary (ascii or binary), a minimal controlDict
and an empty 0/ directory, so the case is readable by the VTK OpenFOAM reader and by `cfd2vec.solvers.polymesh`.

Faces follow the OpenFOAM convention: point order gives the normal pointing out of the owner cell, internal faces
first in upper-triangular order, then boundary patches. `reverse_wall_faces` reverses the point order of the
`building` patch (a normal into the solid), which the owner-based orientation must correct.
"""
import os

import numpy as np

_HEADER = """FoamFile
{{
    version     2.0;
    format      {fmt};
    class       {cls};{arch}
    location    "constant/polyMesh";
    object      {obj};
}}

"""


def _header(fmt, cls, obj, arch="LSB;label=32;scalar=64"):
    a = f'\n    arch        "{arch}";' if fmt == "binary" else ""
    return _HEADER.format(fmt=fmt, cls=cls, obj=obj, arch=a)


def write_case(case_dir, xs, ys, zs, solid, binary=False, reverse_wall_faces=False, label=32, scalar=64,
               order="LSB"):
    """xs, ys, zs: grid lines; solid[i, j, k] marks solid cells (removed). Patches: inlet (x-), outlet (x+),
    sides (y-, y+), top (z+), ground (z-), building (against solid cells). Returns a dict with the cell centres
    (OpenFOAM cell order) and, per patch, the face centres, the into-fluid normals and the owner-cell centres."""
    xs, ys, zs = (np.asarray(a, float) for a in (xs, ys, zs))
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    solid = np.asarray(solid, bool)
    assert solid.shape == (nx, ny, nz)

    def pid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k
    pts = np.array([[xs[i], ys[j], zs[k]] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)])
    cid = -np.ones((nx, ny, nz), np.int64); n = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not solid[i, j, k]:
                    cid[i, j, k] = n; n += 1
    cc = np.zeros((n, 3))
    # face point lists per direction, ordered so the normal points out of the cell (i, j, k)
    def face_pts(i, j, k, d):
        if d == "+x": return [pid(i+1, j, k), pid(i+1, j+1, k), pid(i+1, j+1, k+1), pid(i+1, j, k+1)]
        if d == "-x": return [pid(i, j, k), pid(i, j, k+1), pid(i, j+1, k+1), pid(i, j+1, k)]
        if d == "+y": return [pid(i, j+1, k), pid(i, j+1, k+1), pid(i+1, j+1, k+1), pid(i+1, j+1, k)]
        if d == "-y": return [pid(i, j, k), pid(i+1, j, k), pid(i+1, j, k+1), pid(i, j, k+1)]
        if d == "+z": return [pid(i, j, k+1), pid(i+1, j, k+1), pid(i+1, j+1, k+1), pid(i, j+1, k+1)]
        return [pid(i, j, k), pid(i, j+1, k), pid(i+1, j+1, k), pid(i+1, j, k)]
    steps = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
    outside_patch = {"-x": "inlet", "+x": "outlet", "-y": "sides", "+y": "sides", "+z": "top", "-z": "ground"}
    internal, bnd = [], {p: [] for p in ("inlet", "outlet", "sides", "top", "ground", "building")}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = cid[i, j, k]
                if c < 0:
                    continue
                cc[c] = [(xs[i] + xs[i+1]) / 2, (ys[j] + ys[j+1]) / 2, (zs[k] + zs[k+1]) / 2]
                for d in ("+z", "+y", "+x", "-z", "-y", "-x"):
                    di, dj, dk = steps[d]; ii, jj, kk = i + di, j + dj, k + dk
                    inside = 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz
                    if inside and cid[ii, jj, kk] >= 0:
                        if d[0] == "+":
                            internal.append((c, cid[ii, jj, kk], face_pts(i, j, k, d)))
                    else:
                        bnd["building" if inside else outside_patch[d]].append((c, face_pts(i, j, k, d)))
    internal.sort(key=lambda t: (t[0], t[1]))
    faces, owner, neigh = [f for _, _, f in internal], [o for o, _, _ in internal], [nb for _, nb, _ in internal]
    patches, start = {}, len(faces)
    for name, lst in bnd.items():
        if not lst:
            continue
        for o, f in lst:
            faces.append(f[::-1] if (reverse_wall_faces and name == "building") else f); owner.append(o)
        patches[name] = dict(type="wall" if name in ("ground", "building") else "patch", nFaces=len(lst), startFace=start)
        start += len(lst)
    mesh_dir = os.path.join(case_dir, "constant", "polyMesh"); os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True); os.makedirs(os.path.join(case_dir, "0"), exist_ok=True)
    fmt = "binary" if binary else "ascii"
    arch = f"{order};label={label};scalar={scalar}"
    bo = "<" if order == "LSB" else ">"
    ldt, sdt = f"{bo}i{label // 8}", f"{bo}f{scalar // 8}"
    with open(os.path.join(mesh_dir, "points"), "wb") as f:
        f.write(_header(fmt, "vectorField", "points", arch).encode())
        if binary:
            f.write(f"{len(pts)}\n(".encode()); f.write(pts.astype(sdt).tobytes()); f.write(b")\n")
        else:
            f.write(f"{len(pts)}\n(\n".encode() + "".join(f"({x:.10g} {y:.10g} {z:.10g})\n" for x, y, z in pts).encode() + b")\n")
    with open(os.path.join(mesh_dir, "faces"), "wb") as f:
        if binary:
            off = np.concatenate([[0], np.cumsum([len(x) for x in faces])]).astype(ldt)
            flat = np.concatenate(faces).astype(ldt)
            f.write(_header(fmt, "faceCompactList", "faces", arch).encode())
            f.write(f"{len(off)}\n(".encode()); f.write(off.tobytes()); f.write(b")\n")
            f.write(f"{len(flat)}\n(".encode()); f.write(flat.tobytes()); f.write(b")\n")
        else:
            f.write(_header(fmt, "faceList", "faces").encode())
            f.write(f"{len(faces)}\n(\n".encode() + "".join(f"{len(x)}({' '.join(map(str, x))})\n" for x in faces).encode() + b")\n")
    for name, arr in (("owner", owner), ("neighbour", neigh)):
        with open(os.path.join(mesh_dir, name), "wb") as f:
            f.write(_header(fmt, "labelList", name, arch).encode())
            if binary:
                f.write(f"{len(arr)}\n(".encode()); f.write(np.asarray(arr, ldt).tobytes()); f.write(b")\n")
            else:
                f.write(f"{len(arr)}\n(\n".encode() + "".join(f"{v}\n" for v in arr).encode() + b")\n")
    with open(os.path.join(mesh_dir, "boundary"), "w") as f:
        f.write(_header("ascii", "polyBoundaryMesh", "boundary"))
        f.write(f"{len(patches)}\n(\n")
        for name, p in patches.items():
            f.write(f"    {name}\n    {{\n        type            {p['type']};\n"
                    f"        nFaces          {p['nFaces']};\n        startFace       {p['startFace']};\n    }}\n")
        f.write(")\n")
    with open(os.path.join(case_dir, "system", "controlDict"), "w") as f:
        f.write(_header("ascii", "dictionary", "controlDict").replace('location    "constant/polyMesh"', 'location    "system"'))
        f.write("application foamRun;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\n"
                "writeControl timeStep;\nwriteInterval 1;\n")
    # reference geometry per patch: face centres, into-fluid normals (minus the outward point-order normal), owners
    ref = {}
    for name, lst in bnd.items():
        if not lst:
            continue
        cen, nrm, own = [], [], []
        for o, fp in lst:
            P = pts[fp]; cen.append(P.mean(0))
            nn = np.cross(P[1] - P[0], P[2] - P[0]); nrm.append(-nn / np.linalg.norm(nn)); own.append(cc[o])
        ref[name] = dict(centres=np.array(cen), into_fluid=np.array(nrm), owner_centres=np.array(own))
    return dict(cell_centres=cc, patches=ref, n_faces=len(faces), n_internal=len(internal))


def slab_case(case_dir, binary=False, reverse_wall_faces=False, **kw):
    """The technical review's geometry: fine 0.1 cells below a 0.1-thick slab (z in [0.9, 1.0], x and y in
    [0.1, 0.4]) and one stretched 2.0-tall cell layer above it, so the roof faces are owned by 0.1 x 0.1 x 2 cells
    with centres at z = 2 while the nearest cell centres lie beneath the slab."""
    xs = ys = np.linspace(0.0, 0.5, 6)
    zs = np.concatenate([np.linspace(0.0, 1.0, 11), [3.0]])
    solid = np.zeros((5, 5, 11), bool); solid[1:4, 1:4, 9] = True
    return write_case(case_dir, xs, ys, zs, solid, binary=binary, reverse_wall_faces=reverse_wall_faces, **kw)
