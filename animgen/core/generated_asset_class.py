from pathlib import Path

import numpy as np
import trimesh

from ..io.model_io import load_model

from animgen.render_views import render_views

class GeneratedAssetClass:
    def __init__(self, mesh: str | Path | trimesh.Trimesh):
        if not isinstance(mesh, trimesh.Trimesh):
            self.mesh: trimesh.Trimesh = load_model(mesh)
        else:
            self.mesh: trimesh.Trimesh = mesh
        self.mesh = self._preprocess(self.mesh)
        self.render_config = {
            "yfov_ratio" : 3.0,
            "render_distance" : 2.0,
            "vertical_angles" : [-30, -15, 0, 15, 30],
            "num_views_horizontal" : 12,
            "camera_poses" : None,
            "viewport_size": (1024,1024),
        }      
        self.views, self.depths, self.render_config["camera_poses"] = self.get_views(
            num_views=self.render_config["num_views_horizontal"],
            vertical_angles=self.render_config["vertical_angles"],
            distance=self.render_config["render_distance"],
            yfov_diff_ratio=self.render_config["yfov_ratio"],
            viewport_size=self.render_config['viewport_size'],
        )
        self.views = [np.ascontiguousarray(view) for view in self.views] # Corrects strides
    
    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices
    
    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    def _preprocess(self, mesh: trimesh.Trimesh):
        mesh = mesh.copy()
        center = mesh.vertices.mean(axis=0)
        mesh.vertices -= center
        scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= scale
        return mesh

    def get_views(
        self,
        num_views: int = 8,
        vertical_angles: list[float] = [-30, 0, 30],
        viewport_size: tuple[int, int] = (1024, 1024),
        distance = 2.0,
        yfov_diff_ratio = 3.0,
        debug: bool = False
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        return render_views(mesh = self.mesh, 
                            num_views=num_views, 
                            vertical_angles=vertical_angles, 
                            viewport_size=viewport_size,
                            distance=distance,
                            yfov_diff_ratio=yfov_diff_ratio,
                            debug=debug
                    )