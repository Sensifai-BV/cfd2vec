"""Generic VTK adapter: any VTK-readable volume mesh (vtu, vtk, cgns via pyvista) plus a wall surface (stl / vtp).
Predictions are written back as cell data, which most solvers and pre-processors can import as initial fields."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..schema import Case, Conditioning
from .base import SolverAdapter, footprint_stats, orient_normals_to_fluid, wall_geometry


class VTKAdapter(SolverAdapter):
    name = "vtk"

    def read_case(self, case_dir: str, U_ref: float, L_ref: float, cond: Conditioning, mesh: str = "mesh.vtu",
                  walls: str = "walls.stl", ground_z: Optional[float] = None, case_id: Optional[str] = None) -> Case:
        import pyvista as pv
        m = pv.read(os.path.join(case_dir, mesh))
        pts = np.asarray(m.cell_centers().points, np.float64)
        s = pv.read(os.path.join(case_dir, walls)).extract_surface(algorithm="dataset_surface").compute_normals(
            cell_normals=True, point_normals=False, auto_orient_normals=True)
        fc, fn = np.asarray(s.cell_centers().points), np.asarray(s.cell_data["Normals"])
        fa = np.asarray(s.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"])
        vol = np.abs(np.asarray(m.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"]))
        csz = np.cbrt(np.maximum(vol, 1e-30))
        # a generic VTK volume carries no face-owner connectivity: the surface orientation is kept as read
        # (outward from a closed body after auto-orientation) and reported unverified; nothing is flipped
        fn, orientation = orient_normals_to_fluid(fc, fn)
        orientation["note"] = "surface normals as read (auto-oriented outward from a closed surface); no connectivity"
        if ground_z is not None:                       # add the ground plane as a wall
            d_g = pts[:, 2] - ground_z
        o, H = footprint_stats(fc, fn, fa, None, ground_z)
        L_ref = L_ref if L_ref and L_ref > 0 else H
        d, nrm = wall_geometry(pts, fc, fn)
        if ground_z is not None:
            g = d_g < d
            d = np.where(g, d_g, d).astype(np.float32); nrm[g] = (0.0, 0.0, 1.0)
        cond.has_ground = ground_z is not None
        return Case(case_id=case_id or os.path.basename(os.path.normpath(case_dir)), source="vtk",
                    points=((pts - o) / L_ref).astype(np.float32), wall_dist=d / L_ref, normal=nrm, fields=None,
                    presence=np.zeros(4, bool), cond=cond, L_ref=L_ref, U_ref=U_ref,
                    cell_size=(csz / L_ref).astype(np.float32),
                    meta=dict(case_dir=case_dir, origin=o.tolist(), sampling="full",
                              wall_dist_method="nearest wall-face centre", normal_orientation=orientation))

    def write_initial(self, case_dir: str, fields: dict, mesh: str = "mesh.vtu", out: str = "cfd2vec_initial.vtu"):
        import pyvista as pv
        m = pv.read(os.path.join(case_dir, mesh))
        for k, v in fields.items():
            m.cell_data[k] = np.asarray(v)
        path = os.path.join(case_dir, out); m.save(path)
        return [path]
