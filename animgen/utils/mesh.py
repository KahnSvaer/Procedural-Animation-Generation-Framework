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
