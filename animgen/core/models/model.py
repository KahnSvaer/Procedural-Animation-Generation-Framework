from pathlib import Path
from typing import Any, Optional
import numpy as np
import trimesh

from animgen.core.armature import Armature
from animgen.renderer.renderer import Renderer, render_multiview
from animgen.animation.animator import Animator
from animgen.io.model_input import load_model
from animgen.io.glb_output import export_glb
from animgen.rigging.skinning import compute_auto_skin_weights


class BaseModelClass:
    def __init__(self, mesh: str | Path | trimesh.Trimesh):
        if not isinstance(mesh, trimesh.Trimesh):
            self.mesh: trimesh.Trimesh = load_model(mesh)
        else:
            self.mesh: trimesh.Trimesh = mesh
        self.mesh = self._preprocess(self.mesh)
        self.renderer = self._set_renderer(1024, 1024)
        self.views_output = self._get_views(
            camera_generation_method="dodecahedron",
            renderer_args={
                "return_colored": False,
            },
            sampling_args={
                "radius": 1.5,
            },
        )
        self.adj_graph = self.mesh.face_adjacency

        self.armature: Optional[Armature] = None
        self.animator: Optional[Animator] = None
        self.skin_weights: Optional[dict[str, np.ndarray]] = None

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

    def _set_renderer(
        self, viewport_width: int = 1024, viewport_height: int = 1024
    ) -> Renderer:
        """
        Sets up the renderer with the asset's mesh.
        """
        renderer = Renderer(
            viewport_width=viewport_width, viewport_height=viewport_height
        )
        renderer.set_object(self.mesh)
        renderer.set_camera()
        return renderer

    def _get_views(
        self,
        camera_generation_method: str = "random_sphere",
        renderer_args: dict = {},
        sampling_args: dict = {},
        verbose: bool = True,
    ):
        """
        Generates multiple views of the asset from different camera positions.
        """
        output = render_multiview(
            renderer=self.renderer,
            camera_generation_method=camera_generation_method,
            renderer_args=renderer_args,
            sampling_args=sampling_args,
            verbose=verbose,
        )
        return output

    def compute_skin_weights(self) -> dict[str, np.ndarray]:
        """
        Computes and caches skin weights for the current mesh and armature.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from bone ID to 1D numpy array of vertex skin weights.
        """
        if self.armature is None:
            raise ValueError(
                "Cannot compute skin weights: no Armature assigned to self.armature."
            )
        self.skin_weights = compute_auto_skin_weights(self.mesh, self.armature)
        return self.skin_weights

    def export(
        self,
        output_path: str | Path,
        animation: Optional[Any] = None,
    ) -> Path:
        """
        Exports the model's mesh, armature, and optional animation to a GLB file.
        """
        if self.armature is not None and self.skin_weights is None:
            self.compute_skin_weights()

        return export_glb(
            mesh=self.mesh,
            output_path=output_path,
            armature=self.armature,
            skin_weights=self.skin_weights,
            animation=animation,
        )
