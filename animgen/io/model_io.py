from pathlib import Path
from typing import Union, cast

import trimesh


def load_model(path: Union[str, Path]) -> trimesh.Trimesh:
    asset = trimesh.load(path)

    if isinstance(asset, trimesh.Trimesh):
        return asset

    elif isinstance(asset, trimesh.Scene):
        mesh = asset.dump(concatenate=True)
        if mesh is None:
            raise ValueError("Scene contains no geometries")
        return cast(trimesh.Trimesh, mesh)

    raise ValueError(f"Unsupported mesh type: {type(asset)}")