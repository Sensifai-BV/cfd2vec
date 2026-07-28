"""OpenFOAM adapter (Foundation and ESI lines; ascii or binary).

Reading: reconstructed and decomposed (processor*) cases, through VTK's OpenFOAM reader (pyvista).
Writing initial fields: reconstructed cases only, with foamlib, which keeps every boundaryField entry untouched.
Decomposed cases are rejected before any file changes; write into the reconstructed case, then run decomposePar.
Nothing here modifies the solver or its numerics.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import time
from typing import Optional, Sequence

import numpy as np

from ..schema import Case, Conditioning
from .base import RunReport, SolverAdapter, footprint_stats, orient_normals_to_fluid, wall_geometry

_RES = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+|nan|-?inf)", re.I)
_TIME = re.compile(r"^Time = ([-+]?[0-9.]+(?:[eE][-+]?\d+)?)")


BACKUP_SUFFIX = ".cfd2vec-orig"
BACKUP_MANIFEST = "cfd2vec_backup.json"


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_backup(src: str, dst: str) -> None:
    """Copy src/ to dst/ and record a checksum per file, written last so an interrupted copy is never 'valid'."""
    import json
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    files = {}
    for d, _, names in os.walk(tmp):
        for n in names:
            p = os.path.join(d, n)
            files[os.path.relpath(p, tmp)] = _sha256(p)
    with open(os.path.join(tmp, BACKUP_MANIFEST), "w") as f:
        json.dump(dict(source=os.path.abspath(src), files=files), f, indent=1)
    os.replace(tmp, dst)


def verify_backup(bdir: str) -> dict:
    import json
    mpath = os.path.join(bdir, BACKUP_MANIFEST)
    if not os.path.exists(mpath):
        raise ValueError(f"{bdir}: backup has no manifest")
    man = json.load(open(mpath))
    for rel, h in man["files"].items():
        p = os.path.join(bdir, rel)
        if not os.path.exists(p) or _sha256(p) != h:
            raise ValueError(f"{bdir}: backup file {rel} is missing or modified")
    return man


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")
_CRASH = re.compile(r"(Floating point exception \(core dumped\)|^#\d+\s+Foam::error::printStack|--> FOAM FATAL|^\s*nan\b)",
                    re.M | re.I)


def _foam_file(case_dir: str) -> str:
    f = os.path.join(case_dir, "case.foam")
    if not glob.glob(os.path.join(case_dir, "*.foam")):
        open(f, "w").close()
    return sorted(glob.glob(os.path.join(case_dir, "*.foam")))[0]


def patch_types(case_dir: str) -> dict:
    """patch name -> type from constant/polyMesh/boundary (or processor0 for decomposed cases)."""
    from foamlib import FoamFile
    for p in (os.path.join(case_dir, "constant", "polyMesh", "boundary"),
              os.path.join(case_dir, "processor0", "constant", "polyMesh", "boundary")):
        if os.path.exists(p):
            b = FoamFile(p)[None]
            items = b.items() if isinstance(b, dict) else b
            return {str(k): str(v["type"]) for k, v in items if not str(k).startswith("procBoundary")}
    raise FileNotFoundError(f"no polyMesh/boundary under {case_dir}")


class OpenFOAMAdapter(SolverAdapter):
    name = "openfoam"

    def _reader(self, case_dir: str, decomposed: Optional[bool] = None):
        import pyvista as pv
        if decomposed is None:
            decomposed = not os.path.isdir(os.path.join(case_dir, "constant", "polyMesh")) and \
                bool(glob.glob(os.path.join(case_dir, "processor*")))
        r = pv.POpenFOAMReader(_foam_file(case_dir))
        r.case_type = "decomposed" if decomposed else "reconstructed"
        try:                        # keeps one VTK cell per OpenFOAM cell; recent VTK keeps polyhedra by default
            r.decompose_polyhedra = False
        except AttributeError:
            pass
        r.skip_zero_time = False
        r.cell_to_point_creation = False
        r.enable_all_patch_arrays()
        return r

    def read_case(self, case_dir: str, U_ref: float, L_ref: Optional[float], cond: Conditioning,
                  ground_patches: Sequence[str] = ("ground",), wall_types: Sequence[str] = ("wall",),
                  time: Optional[str] = None, with_fields: bool = False, case_id: Optional[str] = None,
                  origin: Optional[np.ndarray] = None, p_ref: float = 0.0) -> Case:
        """Cell centres, wall distance / normals and cell size of the internal mesh (OpenFOAM cell order).
        With `with_fields`, U, p, k, epsilon of `time` (default: latest) are attached as normalised fields.
        Pressure gauge: incompressible solvers store kinematic pressure; Cp = 2 (p - p_ref) / U_ref^2, where p_ref
        is the reference kinematic pressure of the case (0 for a fixedValue 0 outlet). It is recorded in meta and
        added back by `to_physical`."""
        types = patch_types(case_dir)
        walls = [p for p, t in types.items() if t in wall_types]
        r = self._reader(case_dir)
        if with_fields:
            r.set_active_time_value(float(time) if time else r.time_values[-1])
        else:
            r.set_active_time_value(r.time_values[0])
        mb = r.read()
        internal = mb["internalMesh"]
        pts = np.asarray(internal.cell_centers().points, np.float64)
        vol = np.abs(np.asarray(internal.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"]))
        fc, fn, fa, gm = [], [], [], []
        bnd = mb["boundary"]
        for w in walls:
            s = bnd[w].extract_surface(algorithm="dataset_surface").compute_normals(
                cell_normals=True, point_normals=False, auto_orient_normals=False, consistent_normals=False)
            s = s.compute_cell_sizes(length=False, area=True, volume=False)
            fc.append(np.asarray(s.cell_centers().points)); fa.append(np.asarray(s.cell_data["Area"]))
            fn.append(-np.asarray(s.cell_data["Normals"]))     # OpenFOAM boundary normals point out of the domain
            gm.append(np.full(len(fa[-1]), w in ground_patches))
        csz = np.cbrt(vol)
        # wall normals into the fluid from face-owner connectivity established in the polyMesh files (reader faces
        # matched to polyMesh faces by centre identity, one to one); anything that does not verify keeps the
        # reader's orientation and says so in meta["normal_orientation"]
        owner_centres, note = None, None
        if os.path.isdir(os.path.join(case_dir, "constant", "polyMesh")):
            try:
                from .polymesh import verified_owner_centres
                owner_centres, verification = verified_owner_centres(
                    case_dir, walls, fc, [np.sqrt(np.maximum(a, 0.0)) for a in fa])
            except Exception as e:  # noqa: BLE001  (unparsed mesh file, count mismatch, unmatched face)
                note = f"owner connectivity not verified ({type(e).__name__}: {str(e)[:160]}); reader orientation kept"
        else:
            note = "decomposed layout without a top-level polyMesh: reader orientation kept, unverified"
        fc, fn, fa, gm = map(np.concatenate, (fc, fn, fa, gm))
        fn, orientation = orient_normals_to_fluid(fc, fn, owner_centres)
        if note:
            orientation["note"] = note
        else:
            orientation.update(verification)
        ground_z = float(fc[gm, 2].mean()) if gm.any() else None
        o, H = footprint_stats(fc, fn, fa, gm, ground_z)
        if origin is not None:
            o = np.asarray(origin, np.float64)
        if L_ref is None or L_ref <= 0:          # default characteristic length: mean building / body height
            L_ref = H
        d, nrm = wall_geometry(pts, fc, fn)
        fields, presence = None, np.zeros(4, bool)
        if with_fields:
            cd = internal.cell_data
            U = np.asarray(cd["U"], np.float64); p = np.asarray(cd["p"], np.float64)
            f = np.zeros((len(pts), 6), np.float32)
            f[:, 0:3] = U / U_ref; f[:, 3] = 2 * (p - p_ref) / U_ref ** 2; presence[:2] = True
            if "k" in cd.keys():
                f[:, 4] = np.asarray(cd["k"]) / U_ref ** 2; presence[2] = True
            if "epsilon" in cd.keys():
                f[:, 5] = np.asarray(cd["epsilon"]) * L_ref / U_ref ** 3; presence[3] = True
            fields = f
        cond.has_ground = ground_z is not None
        return Case(case_id=case_id or os.path.basename(os.path.normpath(case_dir)), source="openfoam",
                    points=((pts - o) / L_ref).astype(np.float32), wall_dist=d / L_ref, normal=nrm, fields=fields,
                    presence=presence, cond=cond, L_ref=L_ref, U_ref=U_ref,
                    cell_size=(csz / L_ref).astype(np.float32),
                    meta=dict(case_dir=case_dir, origin=o.tolist(), walls=walls, n_cells=len(pts),
                              ground_z=ground_z, mean_height=H, sampling="full", p_ref=float(p_ref),
                              pressure_gauge="kinematic pressure relative to p_ref",
                              wall_dist_method="nearest wall-face centre", normal_orientation=orientation))

    @staticmethod
    def check_writable_layout(case_dir: str, time: str = "0") -> None:
        """Warm-start writing supports reconstructed (serial) cases only. A case with processor* directories runs
        from the processor-local fields, which this adapter does not write, so it is rejected before any change.
        Recommended: write the initial fields into the reconstructed case, then run decomposePar."""
        procs = sorted(glob.glob(os.path.join(case_dir, "processor*")))
        if procs:
            raise NotImplementedError(f"{case_dir}: decomposed layout ({len(procs)} processor directories); "
                                      "warm-start writing supports reconstructed cases only")
        if not os.path.isdir(os.path.join(case_dir, time)):
            raise FileNotFoundError(f"{case_dir}/{time}: initial-field directory not found")

    @staticmethod
    def backup_dir(case_dir: str, time: str = "0") -> str:
        return os.path.join(case_dir, f"{time}{BACKUP_SUFFIX}")

    def write_initial(self, case_dir: str, fields: dict, time: str = "0", backup: bool = True) -> list:
        """fields: name -> (N,) or (N,3) physical values in OpenFOAM cell order. The existing <time>/ files are kept
        as templates (boundary conditions untouched). Before the first write the original directory is copied to
        <time>.cfd2vec-orig/ together with a checksum manifest; later writes keep that first backup."""
        self.check_writable_layout(case_dir, time)      # layout check needs no foamlib: reject before importing it
        from foamlib import FoamFieldFile
        tdir = os.path.join(case_dir, time)
        for name in fields:                              # validate every target before changing anything
            if not os.path.exists(os.path.join(tdir, name)):
                raise FileNotFoundError(f"{tdir}/{name}: a template field file with boundary conditions is required")
        if backup:
            bdir = self.backup_dir(case_dir, time)
            if os.path.isdir(bdir):
                verify_backup(bdir)
            else:
                make_backup(tdir, bdir)
        written = []
        for name, v in fields.items():
            path = os.path.join(tdir, name)
            ff = FoamFieldFile(path)
            ff.internal_field = np.asarray(v, np.float64)
            written.append(path)
        return written

    def restore_initial(self, case_dir: str, time: str = "0", allow_legacy_backup: bool = False) -> str:
        """Replace <time>/ by the verified backup written by `write_initial`. Raises when no valid backup exists, so
        a fallback can never run 'cold' on predicted fields. `allow_legacy_backup` accepts an unverified
        <time>.orig/ (the pre-manifest backup name, which OpenFOAM tutorials also use for templates)."""
        tdir = os.path.join(case_dir, time)
        bdir = self.backup_dir(case_dir, time)
        if os.path.isdir(bdir):
            verify_backup(bdir)
        elif allow_legacy_backup and os.path.isdir(tdir + ".orig"):
            bdir = tdir + ".orig"
        else:
            raise FileNotFoundError(f"{bdir}: no verified backup of the original initial fields")
        tmp = tdir + ".cfd2vec-restore"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(bdir, tmp, ignore=shutil.ignore_patterns(BACKUP_MANIFEST))
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)
        os.replace(tmp, tdir)
        return bdir

    @staticmethod
    def parse_log(text: str) -> RunReport:
        """Residuals per iteration from a solver log. The first solve of a field in an iteration gives its initial
        residual (the value residualControl tests). Each residual keeps the iteration's `Time = ` value as its
        identifier, so restarted or partial logs can be aligned with monitor output."""
        res: dict = {}
        its, tid = 0, None
        for line in text.splitlines():
            if line.startswith("Time = "):
                its += 1
                t = _TIME.match(line)
                tid = float(t.group(1)) if t else float(its)
            m = _RES.search(line)
            if m and its > 0:
                res.setdefault(m.group(1), {}).setdefault(its, (tid, _float(m.group(2))))
        ids = {k: [v[i][0] for i in sorted(v)] for k, v in res.items()}
        vals = {k: [v[i][1] for i in sorted(v)] for k, v in res.items()}
        ex = re.findall(r"ExecutionTime = ([0-9.]+) s", text)
        conv = "solution converged" in text
        # crash signatures only; the start-up banner "sigFpe : Floating point exception trapping - not supported"
        # is informational and must not count
        crashed = _CRASH.search(text) is not None
        div = crashed or any(not np.isfinite(x) for v in vals.values() for x in v) or \
            any(x > 1e3 for v in vals.values() for x in v[-5:])
        return RunReport(iterations=its, converged=conv, exec_time_s=float(ex[-1]) if ex else float("nan"),
                         residuals=vals, diverged=div, residual_iters=ids)

    @staticmethod
    def _sh(case_dir: str, cmd: str, env_setup: Optional[str]):
        full = f"{env_setup} && {cmd}" if env_setup else cmd
        return subprocess.run(["bash", "-lc", full], cwd=case_dir, capture_output=True, text=True)

    def set_control(self, case_dir: str, entries: dict, env_setup: Optional[str] = None):
        """Set system/controlDict entries with foamDictionary (keeps the file otherwise byte-identical)."""
        for k, v in entries.items():
            r = self._sh(case_dir, f"foamDictionary system/controlDict -entry {k} -set '{v}'", env_setup)
            if r.returncode != 0:
                raise RuntimeError(f"foamDictionary {k}: {r.stderr.strip()[:300]}")

    def run(self, case_dir: str, command: Sequence[str] = ("foamRun",), log_name: str = "log.run",
            env_setup: Optional[str] = None, end_time: Optional[int] = None, continue_run: bool = False) -> RunReport:
        """Run the unchanged solver. `env_setup` is a shell snippet executed first (sourcing the OpenFOAM bashrc).
        `end_time` stops early and writes that time step (probe runs); `continue_run` restarts from latestTime."""
        if end_time is not None:
            self.set_control(case_dir, {"endTime": end_time, "writeControl": "timeStep", "writeInterval": end_time},
                             env_setup)
        if continue_run:
            self.set_control(case_dir, {"startFrom": "latestTime"}, env_setup)
        t0 = time.time()
        r = self._sh(case_dir, " ".join(command) + f" > {log_name} 2>&1", env_setup)
        lp = os.path.join(case_dir, log_name)
        rep = self.parse_log(open(lp).read()) if os.path.exists(lp) else RunReport(0, False, float("nan"))
        rep.log_path = lp
        rep.returncode = int(r.returncode)
        if not np.isfinite(rep.exec_time_s):
            rep.exec_time_s = time.time() - t0
        return rep
