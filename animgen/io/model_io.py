from pathlib import Path
from typing import cast

import trimesh


def load_model(mesh: str | Path | trimesh.Geometry) -> trimesh.Trimesh:
    if isinstance(mesh, (Path, str)):
        mesh = trimesh.load(mesh)

    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    elif isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
        if mesh is None:
            raise ValueError("Scene contains no geometries")
        return cast(trimesh.Trimesh, mesh)
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")
