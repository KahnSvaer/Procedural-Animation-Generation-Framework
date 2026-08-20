"""
Rendering module for animgen. Provides a Renderer class that wraps pyrender and
utilities for rendering multiple views of a 3D mesh. The module supports rendering
of normals, depth maps, face IDs, and barycentric coordinates, as well as generating
camera poses based on random sampling or polyhedral sampling strategies.

References
----------
https://github.com/gtangg12/samesh
"""

import os
import platform as _platform

if _platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault(
        "EGL_DEVICE_ID", "-1"
    )  # NOTE: necessary to not create GPU contention

# for disabling anti-aliasing in pyrender (if needed)
import OpenGL.GL

antialias_active = False
old_gl_enable = OpenGL.GL.glEnable


def new_gl_enable(value):
    if not antialias_active and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)


OpenGL.GL.glEnable = new_gl_enable

# ruff: noqa: E402

import pyrender
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from PIL import Image
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache
import trimesh
from tqdm import tqdm
from typing import Any

from animgen.io.model_input import load_model
from animgen.utils.camera import sample_view_matrices, sample_view_matrices_polyhedra
from animgen.utils.math import range_norm
from animgen.utils.mesh import duplicate_verts
from animgen.renderer.shader_programs import (
    NormalShaderCache,
    FaceidShaderCache,
    BarycentricShaderCache,
)
from animgen.core.types import PoseTransformTensor
from pathlib import Path


def colormap_faces(faces, background=np.array([255, 255, 255])) -> Image.Image:
    """
    Given a face id map, color each face with a random color.
    """
    # print(np.unique(faces, return_counts=True))
    palette = RandomState(0).randint(
        40, 255, (np.max(faces + 2), 3)
    )  # must init every time to get same colors
    # print(palette)
    palette[0] = background
    image = palette[faces + 1, :].astype(np.uint8)  # shift -1 to 0
    return Image.fromarray(image)


def colormap_norms(norms, background=np.array([255, 255, 255])) -> Image.Image:
    """
    Given a normal map, color each normal with a color.
    """
    norms = (norms + 1) / 2
    norms = (norms * 255).astype(np.uint8)
    return Image.fromarray(norms)


DEFAULT_CAMERA_PARAMS = {"fov": 60, "znear": 0.01, "zfar": 16}


class Renderer:
    """ """

    def __init__(self, viewport_width=1024, viewport_height=1024):
        """ """
        self.renderer = pyrender.OffscreenRenderer(
            viewport_height=viewport_height, viewport_width=viewport_width
        )
        self.shaders = {
            "default": DefaultShaderCache(),
            "normals": NormalShaderCache(),
            "faceids": FaceidShaderCache(),
            "barycnt": BarycentricShaderCache(),
        }

    def set_object(self, source: trimesh.Trimesh | str | Path, smooth=False):
        """ """
        source = load_model(source) if isinstance(source, (str, Path)) else source
        self.tmesh = source

        assert isinstance(self.tmesh, trimesh.Trimesh), (
            f"Invalid mesh type {type(self.tmesh)}"
        )
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene.add(pyrender.Mesh.from_trimesh(self.tmesh, smooth=smooth))

        self.tmesh_faceid = duplicate_verts(self.tmesh)
        self.scene_faceid = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene_faceid.add(
            pyrender.Mesh.from_trimesh(self.tmesh_faceid, smooth=smooth)
        )

    def set_camera(self, camera_params: dict | None = None):
        """ """
        self.camera_params = camera_params or dict(DEFAULT_CAMERA_PARAMS)
        self.camera_params["yfov"] = self.camera_params.get(
            "yfov", self.camera_params.pop("fov")
        )
        self.camera_params["yfov"] = self.camera_params["yfov"] * np.pi / 180.0
        self.camera = pyrender.PerspectiveCamera(**self.camera_params)

        self.camera_node = self.scene.add(self.camera)
        self.camera_node_faceid = self.scene_faceid.add(self.camera)

    def render(
        self,
        pose,
        lightdir=np.array([0.0, 0.0, 1.0]),
        return_colored=False,
        interpolate_norms=True,
        blur_matte=False,
    ) -> dict:
        """ """
        self.scene.set_pose(self.camera_node, pose)
        self.scene_faceid.set_pose(self.camera_node_faceid, pose)

        def render_shader(shader: str, scene):
            """ """
            self.renderer._renderer._program_cache = self.shaders[shader]
            return self.renderer.render(scene)

        if return_colored:
            raw_color, raw_depth = render_shader("default", self.scene)
        raw_norms, raw_depth = render_shader("normals", self.scene)
        raw_faces, raw_depth = render_shader("faceids", self.scene_faceid)
        if interpolate_norms:
            raw_bcent, raw_depth = render_shader("barycnt", self.scene_faceid)
        else:
            raw_bcent = None

        def render_norms(norms):
            """ """
            return np.clip((norms / 255.0 - 0.5) * 2, -1, 1)

        def render_depth(depth, offset=2.8, alpha=0.8):
            """ """
            return np.where(
                depth > 0, alpha * (1.0 - range_norm(depth, offset=offset)), 1
            )

        def render_faces(faces):
            """ """
            faces = faces.astype(np.int32)
            faces = faces[:, :, 0] * 65536 + faces[:, :, 1] * 256 + faces[:, :, 2]
            num_faces = self.tmesh_faceid.faces.shape[0]
            faces[faces >= num_faces] = -1  # background + anti-aliasing artifacts
            return faces

        def render_bcent(bcent):
            """ """
            if bcent is None:
                return None
            return np.clip(bcent / 255.0, 0, 1)

        def render_matte(
            norms,
            depth,
            faces,
            bcent,
            alpha=0.5,
            beta=0.25,
            gaussian_kernel_width=5,
            gaussian_sigma=1,
        ):
            """ """
            if interpolate_norms:  # NOTE requires process=True
                verts_index = self.tmesh.faces[faces.reshape(-1)]  # (n, 3)
                v0_norm = self.tmesh.vertex_normals[verts_index[:, 0]]
                v1_norm = self.tmesh.vertex_normals[verts_index[:, 1]]
                v2_norm = self.tmesh.vertex_normals[verts_index[:, 2]]
                bcent_flat = bcent.reshape(-1, 3)
                norms = (
                    v0_norm * bcent_flat[:, 0:1]
                    + v1_norm * bcent_flat[:, 1:2]
                    + v2_norm * bcent_flat[:, 2:3]
                )
                norms = norms.reshape(bcent.shape)

            diffuse = norms @ lightdir
            diffuse = np.clip(diffuse, -1, 1)
            matte = 255 * (diffuse[:, :, None] * alpha + beta)
            matte = np.where(depth[:, :, None] > 0, matte, 255)
            matte = np.clip(matte, 0, 255).astype(np.uint8)
            matte = np.repeat(matte, 3, axis=2)

            if blur_matte:
                matte = (faces == -1)[:, :, None] * matte + (faces != -1)[
                    :, :, None
                ] * cv2.GaussianBlur(
                    matte,
                    (gaussian_kernel_width, gaussian_kernel_width),
                    gaussian_sigma,
                )
            return matte

        norms = render_norms(raw_norms)
        depth = render_depth(raw_depth)
        faces = render_faces(raw_faces)
        bcent = render_bcent(raw_bcent)
        matte = (
            raw_color
            if return_colored
            else render_matte(norms, raw_depth, faces, bcent)
        )  # use original depth for matte

        return {"norms": norms, "depth": depth, "matte": matte, "faces": faces}


def render_multiview(
    renderer: Renderer,
    camera_generation_method="random_sphere",
    renderer_args: dict | None = None,
    sampling_args: dict | None = None,
    lookat_position=np.array([0, 0, 0]),
    verbose=True,
) -> dict[str, Any]:
    """
    Render a set of images from multiple camera viewpoints.

    Camera poses are generated according to the selected sampling strategy,
    rendered sequentially, and grouped by output type. The light direction
    for each view is computed automatically from the camera position toward
    the specified look-at point.

    Args:
        renderer:
            Renderer instance used to render each view.
        camera_generation_method:
            Camera sampling strategy. Must be a key in
            ``POSITION_GENERATORS`` (e.g. ``"random_sphere"`` or a
            supported polyhedral sampling method).
        renderer_args:
            Additional keyword arguments forwarded to
            :meth:`Renderer.render`.
        sampling_args:
            Additional keyword arguments forwarded to the selected camera
            sampling function.
        lookat_position:
            World-space position that all generated cameras face.
        verbose:
            If ``True``, displays a progress bar while rendering.

    Returns:
        A dictionary mapping each render output name (e.g. ``"rgb"``,
        ``"depth"``, ``"matte"``, ``"poses"``) to a list containing that
        output for every sampled viewpoint.
    """
    lookat_position_torch = torch.from_numpy(lookat_position)

    if camera_generation_method == "random_sphere":
        views = sample_view_matrices(
            lookat_position=lookat_position_torch, **sampling_args
        ).numpy()
    else:
        views = sample_view_matrices_polyhedra(
            camera_generation_method,
            lookat_position=lookat_position_torch,
            **sampling_args,
        ).numpy()

    def compute_lightdir(pose: PoseTransformTensor):
        """ """
        lightdir = pose[:3, 3] - (lookat_position)
        return lightdir / np.linalg.norm(lightdir)

    renders = []
    if verbose:
        views = tqdm(views, "Rendering Multiviews...")
    for pose in views:
        outputs = renderer.render(
            pose, lightdir=compute_lightdir(pose), **renderer_args
        )
        outputs["matte"] = Image.fromarray(outputs["matte"])
        outputs["poses"] = pose
        renders.append(outputs)
    return {name: [render[name] for render in renders] for name in renders[0].keys()}
