"""Solver adapters. CFD2vec itself is solver-agnostic; an adapter only has to
    read_case(...)      mesh + walls (+ optional solution) -> canonical Case
    write_initial(...)  predicted fields -> the solver's initial-condition files
    run(...)/parse_log  execute the unchanged solver and report iterations and residuals
Available: openfoam (Foundation and ESI case directories, serial or decomposed), vtk (any VTK-readable mesh plus a
wall surface; output as VTU for solvers that import VTK / CGNS).
"""


def get_adapter(name: str):
    if name == "openfoam":
        from .openfoam import OpenFOAMAdapter
        return OpenFOAMAdapter()
    if name == "vtk":
        from .vtk import VTKAdapter
        return VTKAdapter()
    raise KeyError(name)
