"""
Kinematics utilities for armatures and skeletal hierarchies.

Provides Forward Kinematics (FK) solvers to evaluate posed joint transformations
and global bone rotations, as well as successive chain rotations for procedural animations.
"""

from animgen.core.armature import Armature, Bone
from animgen.core.types import AnimationFrame
from animgen.utils.math import rotation_matrix_from_vectors
import numpy as np
import torch


def compute_forward_kinematics(
    armature: Armature,
    frame: AnimationFrame,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Compute global rotations and posed joint positions for all bones in an armature.

    Traverses the skeletal hierarchy from root bones to leaf bones, propagating
    rotations and translating connected/unconnected child joint positions according
    to the parent's global transformation.

    Parameters
    ----------
    armature : Armature
        The hierarchical armature structure containing bones and parent-child relationships.
    frame : AnimationFrame
        List of local 3x3 rotation matrices for each bone in `armature.bones_list`.

    Returns
    -------
    global_rotations : dict[str, np.ndarray]
        Mapping from bone ID to (3, 3) global rotation matrix in SO(3).
    global_positions : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping from bone ID to a tuple ((3,) posed head position, (3,) posed tail position).
    """
    bone_to_idx = {bone.id: i for i, bone in enumerate(armature.bones_list)}
    global_rotations: dict[str, np.ndarray] = {}
    global_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _eval_bone(bone: Bone) -> None:
        idx = bone_to_idx[bone.id]
        R_local_raw = frame[idx] if idx < len(frame) else np.eye(3)
        if isinstance(R_local_raw, torch.Tensor):
            R_local = R_local_raw.detach().cpu().numpy().astype(np.float64)
        else:
            R_local = np.asarray(R_local_raw, dtype=np.float64)

        h_rest = np.asarray(bone.head, dtype=np.float64)
        t_rest = np.asarray(bone.tail, dtype=np.float64)

        if bone.parent is None:
            R_global = R_local
            h_pose = h_rest.copy()
            t_pose = h_pose + R_global @ (t_rest - h_rest)
        else:
            p_id = bone.parent.id
            if p_id not in global_rotations:
                _eval_bone(bone.parent)
            R_parent_global = global_rotations[p_id]
            h_parent_pose, t_parent_pose = global_positions[p_id]
            h_parent_rest = np.asarray(bone.parent.head, dtype=np.float64)

            R_global = R_parent_global @ R_local

            if bone.is_connected_to_parent:
                h_pose = t_parent_pose.copy()
            else:
                h_pose = h_parent_pose + R_parent_global @ (h_rest - h_parent_rest)

            t_pose = h_pose + R_global @ (t_rest - h_rest)

        global_rotations[bone.id] = R_global
        global_positions[bone.id] = (h_pose, t_pose)

    for root in armature.disconnected_chain_roots:

        def _traverse(b: Bone) -> None:
            _eval_bone(b)
            for c in b.child:
                _traverse(c)

        _traverse(root)

    for bone in armature.bones_list:
        if bone.id not in global_rotations:
            _eval_bone(bone)

    return global_rotations, global_positions


def successive_rotations(
    src: torch.Tensor | np.ndarray,
    tgt: torch.Tensor | np.ndarray,
    is_positions: bool = False,
) -> AnimationFrame:
    """
    Compute successive rotation matrices for a hierarchical chain of segments,
    accounting for the accumulated rotation of parent segments.

    Parameters
    ----------
    src : (N, 3) or (N+1, 3) torch.Tensor | np.ndarray
        The source segment vectors or joint positions.
    tgt : (N, 3) or (N+1, 3) torch.Tensor | np.ndarray
        The target segment vectors or joint positions.
    is_positions : bool, default=False
        If True, inputs are treated as joint positions and diffed along the first dimension
        to produce segment vectors.

    Returns
    -------
    rotations : AnimationFrame
        The local rotation matrices for each segment.
    """
    src_t = torch.as_tensor(src, dtype=torch.float64)
    tgt_t = torch.as_tensor(tgt, dtype=torch.float64)

    if is_positions:
        src_t = torch.diff(src_t, dim=0)
        tgt_t = torch.diff(tgt_t, dim=0)

    N = len(src_t)
    rotations = []
    R_accum = torch.eye(3, dtype=torch.float64, device=src_t.device)

    for i in range(N):
        src_rotated = R_accum @ src_t[i]
        R_local = rotation_matrix_from_vectors(src_rotated, tgt_t[i])
        rotations.append(R_local)
        R_accum = R_local @ R_accum

    return rotations
