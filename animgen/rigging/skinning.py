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

# Alias for backward compatibility
compute_robust_cotangent_laplacian = lambda mesh: compute_cotangent_laplacian(  # noqa: E731
    mesh, return_mass_matrix=True
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
            # Build welded manifold topology for heat diffusion
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
            # Broadcast weights back to original vertex array
            return {
                b_id: welded_weights[b_id][inverse_indices].astype(np.float32)
                for b_id in welded_weights
            }

    # Compute Euclidean distance from each vertex to each bone segment
    bone_dists = np.zeros((N, K), dtype=np.float64)
    for k, bone in enumerate(bones):
        head = np.asarray(bone.head, dtype=np.float64)
        tail = np.asarray(bone.tail, dtype=np.float64)
        dists, _ = dist_point_to_segment_vectorized(mesh.vertices, head, tail)
        bone_dists[:, k] = dists

    # Find closest bone index and distance for each vertex
    min_k = np.argmin(bone_dists, axis=1)
    min_d = np.take_along_axis(bone_dists, min_k[:, None], axis=1).squeeze(axis=1)

    # Boundary heat source matrix P: 1 at nearest bone, 0 elsewhere
    P = np.zeros((N, K), dtype=np.float64)
    np.put_along_axis(P, min_k[:, None], 1.0, axis=1)

    # Dimensionless epsilon based on mesh bounding box size
    if mesh.bounds is not None and len(mesh.vertices) > 1:
        bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    else:
        bbox_diag = 1.0
    eps = max((bbox_diag * 1e-3) ** 2, 1e-8)

    try:
        L, M = compute_robust_cotangent_laplacian(mesh)

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


def compute_auto_skin_weights_blender(
    mesh: trimesh.Trimesh,
    armature: Armature,
) -> dict[str, np.ndarray]:
    """
    Computes automatic skinning weights using Blender's ARMATURE_AUTO operator.
    Requires 'bpy' installed.
    """
    try:
        import bpy
    except ImportError as e:
        raise ImportError(
            "Blender ('bpy') is required for Blender-based skinning weights. "
            "Please ensure bpy is installed in your python environment."
        ) from e

    # Reset / clear scene objects in Blender
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for col in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.objects,
    ):
        for block in list(col):
            if block.users == 0:
                col.remove(block)

    mesh_data = bpy.data.meshes.new("TempSkinMesh")
    mesh_data.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
    mesh_data.update()
    mesh_obj = bpy.data.objects.new("TempSkinMesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    arm_data = bpy.data.armatures.new("TempSkinArmature")
    arm_obj = bpy.data.objects.new("TempSkinArmature", arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = {}
    for bone in armature.bones_list:
        eb = arm_data.edit_bones.new(bone.id)
        eb.head = tuple(bone.head)
        eb.tail = tuple(bone.tail)
        edit_bones[bone.id] = eb

    for bone in armature.bones_list:
        if bone.parent is not None:
            eb = edit_bones[bone.id]
            eb.parent = edit_bones[bone.parent.id]
            eb.use_connect = bone.is_connected_to_parent

    bpy.ops.object.mode_set(mode="OBJECT")

    # Parent mesh to armature using Blender automatic weighting
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except Exception:
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")

    num_verts = len(mesh.vertices)
    bone_ids = [bone.id for bone in armature.bones_list]
    weights_dict = {b_id: np.zeros(num_verts, dtype=np.float32) for b_id in bone_ids}

    vg_idx_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    for v_idx, v in enumerate(mesh_obj.data.vertices):
        for g in v.groups:
            if g.group in vg_idx_to_name:
                b_name = vg_idx_to_name[g.group]
                if b_name in weights_dict:
                    weights_dict[b_name][v_idx] = g.weight

    # Clean up Blender datablocks
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for col in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.objects,
    ):
        for block in list(col):
            if block.users == 0:
                col.remove(block)

    return weights_dict


def compute_auto_skin_weights(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
    use_blender: bool = False,
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
    use_blender : bool
        If True and bpy is installed, computes weights using Blender's ARMATURE_AUTO.
        Defaults to False (pure NumPy/SciPy Bone Heat solver).

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

    if use_blender:
        return compute_auto_skin_weights_blender(mesh, armature)

    return solve_bone_heat_numpy(mesh, armature)


def get_skinning_weight_matrix(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
    use_blender: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Computes automatic skinning weights and returns them as a 2D matrix (num_vertices, num_bones).

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The 3D geometry mesh.
    armature : Armature
        The armature hierarchy.
    use_blender : bool
        If True, use Blender ARMATURE_AUTO backend.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A 2D numpy array of shape (num_vertices, num_bones) with dtype float32,
        and the list of bone IDs corresponding to each column.
    """
    weights_dict = compute_auto_skin_weights(mesh, armature, use_blender=use_blender)
    bone_ids = list(weights_dict.keys())
    matrix = np.column_stack([weights_dict[b_id] for b_id in bone_ids])
    return matrix, bone_ids
