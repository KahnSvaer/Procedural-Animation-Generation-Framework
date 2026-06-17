from pathlib import Path

import numpy as np
import trimesh
from typing import List, Union

from io.model_io import load_model

class GeneratedAssetClass:
    def __init__(self, path: Union[str, Path]):
        self.mesh = load_model(path)
        self._preprocess()
    
    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices
    
    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    def _preprocess(self):
        self.mesh.vertices -= self.mesh.center_mass # Center the mesh at the origin
        scale = 1.0 / np.max(np.linalg.norm(self.mesh.vertices, axis=1)) # Scale the mesh to fit within a unit sphere
        self.mesh.vertices *= scale