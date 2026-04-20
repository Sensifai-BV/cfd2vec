"""CFD2vec: a self-supervised, mesh-agnostic foundation model for 3-D steady and time-averaged flows.

The core is independent of any solver or dataset:
    inputs   Geometry (wall distance, normals, height), Mesh (query points, local cell size),
             Boundary conditions (inflow direction, profile, turbulence, ground), Physics (Re, closure),
             optional prior field (a low-fidelity solution of the same problem)
    outputs  a learned representation (tokens + global embedding) and an implicit field decoder
             that evaluates U, Cp, k, epsilon at arbitrary points.
Data sources (cfd2vec.sources) and solvers (cfd2vec.solvers) are plug-ins.
Downstream tasks (cfd2vec.tasks): field prediction, super-fidelity / solver warm start, embedding + retrieval.
"""
__version__ = "0.1.0.dev0"

from .schema import Case, Conditioning, CLOSURES, FIELDS, GROUNDS  # noqa: F401


def __getattr__(name):  # lazy import keeps `import cfd2vec` torch-free for data tooling
    if name == "CFD2vec":
        from .api import CFD2vec
        return CFD2vec
    raise AttributeError(name)
