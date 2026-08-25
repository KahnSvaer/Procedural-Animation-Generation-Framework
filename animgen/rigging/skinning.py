"""
This module provides automatic skinning weight computation using a pure
NumPy/SciPy Bone Heat Weighting algorithm (based on Baran & Popović 2007
and Blender's mesh Laplacian skinning).

It computes smooth, geometry-aware per-vertex bone influences given a
trimesh.Trimesh and an Armature without requiring Blender ('bpy') at runtime.

References
----------
.. [1] Baran, I., & Popović, J. (2007). Automatic rigging and animation of
       3D characters. ACM Transactions on Graphics (TOG), 26(3), 72-es.
       https://doi.org/10.1145/1276377.1276467
.. [2] Blender Foundation. "Armature Deform: Automatic Weights (meshlaplacian.cc)".
       Blender Kernel / Armature Editors.
       https://projects.blender.org/blender/blender/src/branch/main/source/blender/editors/armature/meshlaplacian.cc
.. [3] Pinkall, U., & Polthier, K. (1993). Computing discrete minimal surfaces
       and their conjugates. Experimental Mathematics, 2(1), 15-36.
       https://doi.org/10.1080/10586458.1993.10504266
"""

from typing import Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

from animgen.core.armature import Armature
from animgen.utils.mesh import (
    compute_cotangent_laplacian,
    dist_point_to_segment_vectorized,
)


def solve_bone_heat_numpy(
    mesh: trimesh.Trimesh,
    armature: Armature,
    heat_falloff_scale: float = 1.0,
    weld_tolerance: float = 1e-5,
) -> dict[str, np.ndarray]:
    """
    Computes automatic skinning weights using the Bone Heat diffusion solver
    (Baran & Popović 2007 Pinocchio formulation / Blender meshlaplacian).

    Automatically handles dirty geometry and duplicate vertices (e.g. UV seams,
    split normals) by solving on a spatially welded topological graph and
    broadcasting weights back to preserve original vertex arrays and textures.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The 3D model geometry.
    armature : Armature
        The armature hierarchy.
    heat_falloff_scale : float
        Scaling factor for heat conduction.
    weld_tolerance : float
        Spatial tolerance for welding coincident vertices along seams.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from bone id to 1D numpy array of float32 skin weights of shape (N,).
    """
    N = len(mesh.vertices)
    bones = armature.bones_list
    K = len(bones)

    if N == 0 or K == 0:
        return {}

    # Check for duplicate coincident vertices (common in textured models with UV seams)
    if len(mesh.faces) > 0 and weld_tolerance > 0:
        digits = int(max(0, -np.log10(weld_tolerance)))
        rounded_verts = np.round(mesh.vertices, decimals=digits)
        unique_verts, inverse_indices = np.unique(
            rounded_verts, axis=0, return_inverse=True
        )
        if len(unique_verts) < N:
            welded_faces = inverse_indices[mesh.faces]
            valid_faces = (
                (welded_faces[:, 0] != welded_faces[:, 1])
                & (welded_faces[:, 1] != welded_faces[:, 2])
                & (welded_faces[:, 2] != welded_faces[:, 0])
            )
            welded_mesh = trimesh.Trimesh(
                vertices=unique_verts,
                faces=welded_faces[valid_faces],
                process=False,
            )
            welded_weights = solve_bone_heat_numpy(
                welded_mesh,
                armature,
                heat_falloff_scale=heat_falloff_scale,
                weld_tolerance=0.0,
            )
            return {
                b_id: welded_weights[b_id][inverse_indices].astype(np.float32)
                for b_id in welded_weights
            }

    bone_dists = np.zeros((N, K), dtype=np.float64)
    for k, bone in enumerate(bones):
        head = np.asarray(bone.head, dtype=np.float64)
        tail = np.asarray(bone.tail, dtype=np.float64)
        dists, _ = dist_point_to_segment_vectorized(mesh.vertices, head, tail)
        bone_dists[:, k] = dists

    min_k = np.argmin(bone_dists, axis=1)
    min_d = np.take_along_axis(bone_dists, min_k[:, None], axis=1).squeeze(axis=1)

    P = np.zeros((N, K), dtype=np.float64)
    np.put_along_axis(P, min_k[:, None], 1.0, axis=1)

    if mesh.bounds is not None and len(mesh.vertices) > 1:
        bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    else:
        bbox_diag = 1.0
    eps = max((bbox_diag * 1e-3) ** 2, 1e-8)

    try:
        L, M = compute_cotangent_laplacian(mesh, return_mass_matrix=True)

        # Pinocchio / Blender heat conduction operator:
        # H_ii = heat_falloff_scale / max(min_d(v)^2, eps)
        H_diag = heat_falloff_scale / np.maximum(min_d**2, eps)
        MH_diag = M.diagonal() * H_diag
        MH = sp.diags(MH_diag, 0, shape=(N, N), format="csr")

        A = L + MH
        rhs = MH @ P

        solver = spla.factorized(A.tocsc())
        W = solver(rhs)
    except Exception:
        # Fallback to inverse distance weighting if Laplacian fails
        inv_d = 1.0 / np.maximum(bone_dists, 1e-6) ** 2
        W = inv_d / np.sum(inv_d, axis=1, keepdims=True)

    # Post-process: clamp negatives, handle zero-weight rows, and normalize
    W = np.maximum(0.0, W)
    W_sum = np.sum(W, axis=1, keepdims=True)
    zero_mask = W_sum.squeeze(axis=-1) < 1e-8
    if np.any(zero_mask):
        W[zero_mask, :] = 0.0
        W[zero_mask, min_k[zero_mask]] = 1.0
        W_sum = np.sum(W, axis=1, keepdims=True)

    W = W / W_sum

    return {bones[k].id: W[:, k].astype(np.float32) for k in range(K)}


def compute_auto_skin_weights(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
) -> dict[str, np.ndarray]:
    """
    Computes automatic skinning weights for each vertex and bone using
    the Bone Heat diffusion engine.

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The 3D model geometry.
    armature : Armature
        The armature hierarchy.

    Returns
    -------
    dict[str, np.ndarray]
        A mapping from bone id to a 1D numpy array of shape (N,) containing the
        skin weights for each of the N vertices.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh of type trimesh.Trimesh, got {type(mesh)}")

    if armature is None or not armature.bones_list:
        raise ValueError(
            "An Armature with at least one bone must be provided for skinning."
        )

    return solve_bone_heat_numpy(mesh, armature)


def get_skinning_weight_matrix(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
) -> tuple[np.ndarray, list[str]]:
    """
    Computes automatic skinning weights and returns them as a 2D matrix (num_vertices, num_bones).

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The 3D geometry mesh.
    armature : Armature
        The armature hierarchy.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A 2D numpy array of shape (num_vertices, num_bones) with dtype float32,
        and the list of bone IDs corresponding to each column.
    """
    weights_dict = compute_auto_skin_weights(mesh, armature)
    bone_ids = list(weights_dict.keys())
    matrix = np.column_stack([weights_dict[b_id] for b_id in bone_ids])
    return matrix, bone_ids
