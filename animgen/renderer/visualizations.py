import trimesh
import numpy as np
from typing import Any, Tuple, TYPE_CHECKING
from pathlib import Path
from animgen.io.model_io import load_model

if TYPE_CHECKING:
    from animgen.core.models.model import BaseModelClass


def visualize_skeleton(
    skeleton: Any,
    skeleton_edges: np.ndarray | None = None,
    radius: float = 0.01,
    color_bones: Tuple[int, int, int, int] | list = [230, 30, 30, 255],
    color_joints: Tuple[int, int, int, int] | list = [30, 30, 230, 255],
) -> trimesh.Scene:
    """
    Create a 3D scene displaying only the standalone skeleton (as solid cylinders around bone edges
    and solid spheres around joint vertex points).

    Parameters
    ----------
    skeleton : trimesh.path.Path3D | np.ndarray | tuple
        Either a trimesh Path3D object, a (skeleton_vertices, skeleton_edges) tuple,
        or an (N, 3) numpy array of joint positions.
    skeleton_edges : np.ndarray, optional
        Indices of connected skeleton edges if skeleton is passed as a vertex array.
    radius : float
        Radius of the bone cylinders.
    color_bones : list or tuple
        RGBA color for the bone cylinders. Default is red [230, 30, 30, 255].
    color_joints : list or tuple
        RGBA color for the joint spheres. Default is blue [30, 30, 230, 255].

    Returns
    -------
    scene : trimesh.Scene
        3D scene containing solid bone cylinders and joint spheres.
    """
    # Unpack skeleton vertices and edges robustly
    skeleton_vertices = None
    if isinstance(skeleton, trimesh.path.Path3D):
        skeleton_vertices = skeleton.vertices
        edges_list = []
        for entity in skeleton.entities:
            if hasattr(entity, "nodes"):
                edges_list.append(entity.nodes)
        skeleton_edges = edges_list
    elif (
        isinstance(skeleton, (tuple, list))
        and len(skeleton) == 2
        and isinstance(skeleton[0], np.ndarray)
    ):
        skeleton_vertices = skeleton[0]
        skeleton_edges = skeleton[1]
    elif isinstance(skeleton, np.ndarray):
        skeleton_vertices = skeleton

    cylinders = []
    if skeleton_vertices is not None and skeleton_edges is not None:
        for edge in skeleton_edges:
            edge_seq = np.atleast_1d(edge)
            if len(edge_seq) < 2:
                continue
            for idx in range(len(edge_seq) - 1):
                u, v = edge_seq[idx], edge_seq[idx + 1]
                seg = [skeleton_vertices[u], skeleton_vertices[v]]
                try:
                    cyl = trimesh.creation.cylinder(
                        radius=radius, segment=seg, sections=8
                    )
                    cyl.visual.face_colors = color_bones
                    cylinders.append(cyl)
                except Exception:
                    pass

    spheres = []
    if skeleton_vertices is not None:
        for pt in skeleton_vertices:
            try:
                sph = trimesh.creation.icosphere(radius=radius * 1.5, subdivisions=1)
                sph.vertices += pt
                sph.visual.face_colors = color_joints
                spheres.append(sph)
            except Exception:
                pass

    return trimesh.Scene(cylinders + spheres)


def visualize_skeleton_over_mesh(
    source: "trimesh.Trimesh | BaseModelClass | str | Path",
    skeleton: Any = None,
    skeleton_edges: np.ndarray | None = None,
    radius: float = 0.01,
    opacity: float = 0.25,
) -> trimesh.Scene:
    """
    Create a 3D scene displaying the skeleton (as solid red cylinders and blue joint spheres)
    overlaid on the original mesh rendered with customizable low opacity for clear viewing.

    Parameters
    ----------
    source : trimesh.Trimesh | BaseModelClass | str | Path
        The input 3D mesh model instance or object.
    skeleton : trimesh.path.Path3D | np.ndarray | tuple, optional
        Either a trimesh Path3D object, a (skeleton_vertices, skeleton_edges) tuple,
        or an (N, 3) numpy array of joint positions.
    skeleton_edges : np.ndarray, optional
        Indices of connected skeleton edges if skeleton is passed as a vertex array.
    radius : float
        Radius of the bone cylinders.
    opacity : float
        Opacity of the outer mesh (0.0 = completely invisible, 1.0 = opaque). Default is 0.25.

    Returns
    -------
    scene : trimesh.Scene
        3D scene containing translucent mesh, solid bones, and joint spheres.
    """
    if hasattr(source, "mesh"):
        mesh = source.mesh
    else:
        mesh = source

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = load_model(mesh)

    # 1. Low-opacity copy of original mesh for clear viewing
    vis_mesh = mesh.copy()
    alpha_int = int(np.clip(opacity, 0.0, 1.0) * 255)

    # Dedicated transparent material for OpenGL / Pyglet / Trimesh rendering
    try:
        mat = trimesh.visual.material.SimpleMaterial(
            diffuse=[200, 200, 200, alpha_int], ambient=[100, 100, 100, alpha_int]
        )
        vis_mesh.visual.material = mat
    except Exception:
        pass

    vis_mesh.visual.face_colors = [200, 200, 200, alpha_int]

    # Build standalone skeleton 3D scene
    skel_scene = visualize_skeleton(
        skeleton=skeleton,
        skeleton_edges=skeleton_edges,
        radius=radius,
    )

    # Combine mesh and skeleton geometry
    return trimesh.Scene([vis_mesh] + list(skel_scene.geometry.values()))
