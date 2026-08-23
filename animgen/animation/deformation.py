"""
Mesh deformation and skinning algorithms for skeletal animation playback.

Provides Dual Quaternion Skinning (DQS) to prevent volume loss and candy-wrapper
twisting artifacts, alongside Linear Blend Skinning (LBS) for classic mesh morphing.
"""

from typing import Literal, Union
from animgen.core.armature import Armature
from animgen.utils.math import rotation_matrix_to_quaternion
import numpy as np
import trimesh


def apply_dual_quaternion_skinning_deformation(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
    global_bone_rotations: dict[str, np.ndarray],
    global_bone_heads: dict[str, np.ndarray],
    skin_weights: dict[str, np.ndarray],
) -> trimesh.Trimesh:
    """
    Apply Dual Quaternion Skinning (DQS) to deform a mesh based on posed bone transformations.

    Dual Quaternion Linear Blending (DLB) preserves volume during large joint rotations
    and twisting, completely eliminating the 'candy-wrapper' pinch artifacts of LBS.

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The original reference mesh geometry in rest pose.
    armature : Armature
        The armature hierarchy containing bone rest positions.
    global_bone_rotations : dict[str, np.ndarray]
        Mapping from bone ID to (3, 3) global rotation matrix in SO(3).
    global_bone_heads : dict[str, np.ndarray]
        Mapping from bone ID to (3,) global posed head position.
    skin_weights : dict[str, np.ndarray]
        Mapping from bone ID to 1D array of per-vertex skinning weights of shape (N,).

    Returns
    -------
    deformed_mesh : trimesh.Trimesh
        A new deformed trimesh.Trimesh geometry with preserved volume and details.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    v_rest = np.asarray(mesh.vertices, dtype=np.float64)
    num_verts = len(v_rest)
    bones = armature.bones_list
    num_bones = len(bones)

    if num_bones == 0:
        return mesh.copy()

    q0_list = []
    qd_list = []
    w_matrix_cols = []

    pivot_q0 = None
    for bone in bones:
        b_id = bone.id
        w = np.asarray(skin_weights.get(b_id, np.zeros(num_verts)), dtype=np.float64)
        w_matrix_cols.append(w)

        h_rest = np.asarray(bone.head, dtype=np.float64)
        h_pose = global_bone_heads.get(b_id, h_rest)
        R_global = global_bone_rotations.get(b_id, np.eye(3, dtype=np.float64))

        t_vec = h_pose - R_global @ h_rest

        q0 = rotation_matrix_to_quaternion(R_global)

        if pivot_q0 is None:
            pivot_q0 = q0
        elif np.dot(q0, pivot_q0) < 0.0:
            q0 = -q0

        w0 = q0[0]
        v0 = q0[1:4]

        # Dual quaternion part: qd = 0.5 * ( [0, t_vec] * q0 )
        wd = -0.5 * float(np.dot(t_vec, v0))
        vd = 0.5 * (w0 * t_vec + np.cross(t_vec, v0))
        qd = np.array([wd, vd[0], vd[1], vd[2]], dtype=np.float64)

        q0_list.append(q0)
        qd_list.append(qd)

    Q0_bones = np.array(q0_list, dtype=np.float64)
    Qd_bones = np.array(qd_list, dtype=np.float64)

    W = np.column_stack(w_matrix_cols)
    total_w = np.sum(W, axis=1, keepdims=True)

    unweighted = total_w[:, 0] < 1e-8
    total_w[unweighted] = 1.0
    W_norm = W / total_w

    Q0_blend = W_norm @ Q0_bones
    Qd_blend = W_norm @ Qd_bones

    q0_len = np.linalg.norm(Q0_blend, axis=1, keepdims=True)
    q0_len = np.where(q0_len < 1e-12, 1.0, q0_len)

    Q0_norm = Q0_blend / q0_len
    Qd_norm = Qd_blend / q0_len

    w0 = Q0_norm[:, 0:1]
    v0 = Q0_norm[:, 1:4]
    wd = Qd_norm[:, 0:1]
    vd = Qd_norm[:, 1:4]

    # Blended translation: t_blend = 2.0 * (w0 * vd - wd * v0 + (v0 x vd))
    t_blend = 2.0 * (w0 * vd - wd * v0 + np.cross(v0, vd))

    # v_rot = v_rest + 2.0 * w0 * (v0 x v_rest) + 2.0 * (v0 x (v0 x v_rest))
    v0_cross_v = np.cross(v0, v_rest)
    v_rot = v_rest + 2.0 * w0 * v0_cross_v + 2.0 * np.cross(v0, v0_cross_v)

    v_deformed = v_rot + t_blend
    if np.any(unweighted):
        v_deformed[unweighted] = v_rest[unweighted]

    deformed_mesh = mesh.copy()
    deformed_mesh.vertices = v_deformed
    return deformed_mesh


def apply_linear_blend_skinning_deformation(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
    global_bone_rotations: dict[str, np.ndarray],
    global_bone_heads: dict[str, np.ndarray],
    skin_weights: dict[str, np.ndarray],
) -> trimesh.Trimesh:
    """
    Apply Linear Blend Skinning (LBS) to deform a mesh based on posed bone transformations.

    Evaluates the weighted sum of bone transformations on each vertex:
        v' = sum_b w_{v, b} * (h_{pose, b} + R_{global, b} * (v_{rest} - h_{rest, b}))

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The original reference mesh geometry in rest pose.
    armature : Armature
        The armature hierarchy containing bone rest positions and parent-child relationships.
    global_bone_rotations : dict[str, np.ndarray]
        Mapping from bone ID to (3, 3) global rotation matrix in SO(3).
    global_bone_heads : dict[str, np.ndarray]
        Mapping from bone ID to (3,) global posed head position.
    skin_weights : dict[str, np.ndarray]
        Mapping from bone ID to 1D array of per-vertex skinning weights of shape (N,).

    Returns
    -------
    deformed_mesh : trimesh.Trimesh
        A new deformed trimesh.Trimesh geometry preserving original faces and visuals.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    v_rest = np.asarray(mesh.vertices, dtype=np.float64)
    num_verts = len(v_rest)
    v_deformed = np.zeros((num_verts, 3), dtype=np.float64)
    total_weight = np.zeros(num_verts, dtype=np.float64)

    for bone in armature.bones_list:
        b_id = bone.id
        if b_id not in skin_weights:
            continue
        w = np.asarray(skin_weights[b_id], dtype=np.float64)
        if not np.any(w > 0):
            continue

        h_rest = np.asarray(bone.head, dtype=np.float64)
        h_pose = global_bone_heads.get(b_id, h_rest)
        R_global = global_bone_rotations.get(b_id, np.eye(3, dtype=np.float64))

        v_b = h_pose + (v_rest - h_rest) @ R_global.T
        v_deformed += w[:, np.newaxis] * v_b
        total_weight += w

    unweighted = total_weight < 1e-8
    if np.any(unweighted):
        v_deformed[unweighted] = v_rest[unweighted]
        total_weight[unweighted] = 1.0

    valid_mask = ~unweighted
    v_deformed[valid_mask] /= total_weight[valid_mask, np.newaxis]

    deformed_mesh = mesh.copy()
    deformed_mesh.vertices = v_deformed
    return deformed_mesh


def apply_mesh_deformation(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
    global_bone_rotations: dict[str, np.ndarray],
    global_bone_heads: dict[str, np.ndarray],
    skin_weights: dict[str, np.ndarray],
    method: Literal["dqs", "lbs"] = "dqs",
) -> trimesh.Trimesh:
    """
    Deform a mesh surface according to posed bone transformations and skinning weights.

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The reference 3D mesh geometry in rest pose.
    armature : Armature
        The armature hierarchy.
    global_bone_rotations : dict[str, np.ndarray]
        Mapping from bone ID to (3, 3) global rotation matrix in SO(3).
    global_bone_heads : dict[str, np.ndarray]
        Mapping from bone ID to (3,) global posed head position.
    skin_weights : dict[str, np.ndarray]
        Mapping from bone ID to 1D array of per-vertex skinning weights.
    method : {"dqs", "lbs"}, default="dqs"
        Skinning deformation method:
        - "dqs": Dual Quaternion Skinning (prevents volume loss and candy-wrapper artifacts).
        - "lbs": Linear Blend Skinning.

    Returns
    -------
    deformed_mesh : trimesh.Trimesh
        The deformed trimesh.Trimesh geometry.
    """
    if method == "dqs":
        return apply_dual_quaternion_skinning_deformation(
            mesh=mesh,
            armature=armature,
            global_bone_rotations=global_bone_rotations,
            global_bone_heads=global_bone_heads,
            skin_weights=skin_weights,
        )
    elif method == "lbs":
        return apply_linear_blend_skinning_deformation(
            mesh=mesh,
            armature=armature,
            global_bone_rotations=global_bone_rotations,
            global_bone_heads=global_bone_heads,
            skin_weights=skin_weights,
        )
    else:
        raise ValueError(
            f"Unknown skinning method '{method}'. Expected 'dqs' or 'lbs'."
        )
