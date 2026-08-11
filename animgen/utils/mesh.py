import trimesh
import numpy as np


def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center
    return mesh


def duplicate_verts(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Call before coloring mesh to avoid face interpolation since openGL stores color attributes per vertex.

        ...
        mesh = duplicate_verts(mesh)
        mesh.visual.face_colors = colors
        ...

    NOTE: removes visuals for verticies, but preserves for faces.
    """
    verts = mesh.vertices[mesh.faces.reshape(-1), :]
    faces = np.arange(0, verts.shape[0])
    faces = faces.reshape(-1, 3)
    try:
        face_colors = mesh.visual.face_colors
    except (AttributeError, ValueError, IndexError):
        face_colors = np.full(
            (len(mesh.faces), 4), [200, 200, 200, 255], dtype=np.uint8
        )
    return trimesh.Trimesh(
        vertices=verts, faces=faces, face_colors=face_colors, process=False
    )


def taubin_smoothing(
    mesh: trimesh.Trimesh,
    lamb: float = 0.5,
    nu: float = 0.53,
    iterations: int = 10,
) -> trimesh.Trimesh:
    """
    Smooth a mesh using Taubin filtering (Laplacian smoothing with shrinkage compensation).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to be smoothed.
    lamb : float
        The shrinkage factor (0.0 < lamb < 1.0).
    nu : float
        The dilation factor (0.0 < nu < 1.0, typically nu > lamb).
    iterations : int
        The number of smoothing iterations.

    Returns
    -------
    smoothed_mesh : trimesh.Trimesh
        A smoothed copy of the input mesh.
    """
    mesh_copy = mesh.copy()
    trimesh.smoothing.filter_taubin(mesh_copy, lamb=lamb, nu=nu, iterations=iterations)
    return mesh_copy
