from pathlib import Path

import numpy as np
import trimesh

from animgen.renderer.renderer import Renderer, render_multiview
from animgen.io.model_io import load_model

class GeneratedAssetClass:
    def __init__(self, mesh: str | Path | trimesh.Trimesh):
        if not isinstance(mesh, trimesh.Trimesh):
            self.mesh: trimesh.Trimesh = load_model(mesh)
        else:
            self.mesh: trimesh.Trimesh = mesh
        self.mesh = self._preprocess(self.mesh)
        self.renderer = self._set_renderer(1024, 1024)
        self.views_output = self._get_views(
            camera_generation_method='dodecahedron',
            renderer_args={
                'return_colored': False,
            },
            sampling_args={
                'radius': 1.5,
            }
        )
        self.adj_graph = self.mesh.face_adjacency

    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices
    
    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    def _preprocess(self, mesh: trimesh.Trimesh):
        """
        Preprocesses the mesh for rendering.
        """
        mesh = mesh.copy()
        center = mesh.vertices.mean(axis=0)
        mesh.vertices -= center
        scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= scale
        return mesh
    
    def _set_renderer(self, viewport_width: int = 1024, viewport_height: int = 1024) -> Renderer:
        """
        Sets up the renderer with the asset's mesh.
        """
        renderer = Renderer(viewport_width=viewport_width, viewport_height=viewport_height)
        renderer.set_object(self.mesh)
        renderer.set_camera()
        return renderer

    def _get_views(
        self,
        camera_generation_method: str = 'random_sphere',
        renderer_args: dict = {},
        sampling_args: dict = {},
        verbose: bool = True
    ):
        """
        Generates multiple views of the asset from different camera positions.
        """
        output = render_multiview(
            renderer = self.renderer,
            camera_generation_method = camera_generation_method,
            renderer_args = renderer_args,
            sampling_args = sampling_args,
            verbose = verbose
        )
        return output