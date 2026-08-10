import numpy as np
import trimesh
import torch
import bpy
import mathutils
from typing import Union, List, Any

from animgen.core.spline import Spline


def build_bishop_frame(
    spine_pts_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Bishop parallel transport frames along a 3D spline/spine path.

    Parameters
    ----------
    spine_pts_np : (N, 3) ndarray
        Dense sequence of 3D points representing the curve.

    Returns
    -------
    T_seg : (N-1, 3) ndarray
        Tangent vectors for each segment.
    N_seg : (N-1, 3) ndarray
        Normal vectors for each segment.
    B_seg : (N-1, 3) ndarray
        Binormal vectors for each segment.
    segment_lengths : (N-1,) ndarray
        Lengths of each segment.
    s : (N,) ndarray
        Cumulative arc lengths at each point along the spine.
    """
    N_points = len(spine_pts_np)
    T_seg = np.zeros((N_points - 1, 3))
    N_seg = np.zeros((N_points - 1, 3))
    B_seg = np.zeros((N_points - 1, 3))

    for i in range(N_points - 1):
        diff = spine_pts_np[i + 1] - spine_pts_np[i]
        length = np.linalg.norm(diff)
        T_seg[i] = diff / max(length, 1e-12)

    T0 = T_seg[0]
    if abs(T0[0]) < 0.9:
        V = np.array([1.0, 0.0, 0.0])
    else:
        V = np.array([0.0, 1.0, 0.0])
    N0 = V - np.dot(V, T0) * T0
    N0_norm = np.linalg.norm(N0)
    N0 = N0 / max(N0_norm, 1e-12)
    B0 = np.cross(T0, N0)

    N_seg[0] = N0
    B_seg[0] = B0

    for i in range(1, N_points - 1):
        t_prev = T_seg[i - 1]
        t_curr = T_seg[i]
        n_prev = N_seg[i - 1]

        axis_rot = np.cross(t_prev, t_curr)
        axis_norm = np.linalg.norm(axis_rot)
        if axis_norm < 1e-8:
            n_curr = n_prev - np.dot(n_prev, t_curr) * t_curr
            n_curr_norm = np.linalg.norm(n_curr)
            n_curr = n_curr / max(n_curr_norm, 1e-12)
        else:
            axis_rot = axis_rot / axis_norm
            dot_val = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            theta = np.arccos(dot_val)
            n_curr = (
                n_prev * np.cos(theta)
                + np.cross(axis_rot, n_prev) * np.sin(theta)
                + axis_rot * np.dot(axis_rot, n_prev) * (1.0 - np.cos(theta))
            )
            n_curr = n_curr - np.dot(n_curr, t_curr) * t_curr
            n_curr_norm = np.linalg.norm(n_curr)
            n_curr = n_curr / max(n_curr_norm, 1e-12)

        N_seg[i] = n_curr
        B_seg[i] = np.cross(t_curr, n_curr)

    diffs = np.diff(spine_pts_np, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    return T_seg, N_seg, B_seg, segment_lengths, s


def deform_mesh_to_spine_numpy(
    mesh: trimesh.Trimesh,
    source_spine: np.ndarray,
    target_spine: np.ndarray,
    chunk_size: int = 10000,
) -> trimesh.Trimesh:
    """
    Deform a mesh using Bishop frame coordinate projection in pure NumPy.
    This method is geometrically exact and volume-preserving.
    """
    # 1. Build Bishop parallel transport frames along both spines
    T_src, N_src, B_src, _, _ = build_bishop_frame(source_spine)
    T_tgt, N_tgt, B_tgt, _, _ = build_bishop_frame(target_spine)

    A_src = source_spine[:-1]
    D_src = source_spine[1:] - source_spine[:-1]
    L2_src = np.sum(D_src**2, axis=1)
    L2_src = np.maximum(L2_src, 1e-12)

    A_tgt = target_spine[:-1]
    D_tgt = target_spine[1:] - target_spine[:-1]

    N_points = len(source_spine)
    V_new = np.zeros_like(mesh.vertices)
    num_vertices = len(mesh.vertices)

    # 2. Map coordinates chunk by chunk to limit memory footprints
    for start_idx in range(0, num_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_vertices)
        V_chunk = mesh.vertices[start_idx:end_idx]

        # Calculate squared distance from all vertices in the chunk to all segments
        disp = V_chunk[:, None, :] - A_src[None, :, :]
        dot = np.sum(disp * D_src[None, :, :], axis=2)
        t_val = np.clip(dot / L2_src[None, :], 0.0, 1.0)
        proj = A_src[None, :, :] + t_val[:, :, None] * D_src[None, :, :]
        dist2 = np.sum((V_chunk[:, None, :] - proj) ** 2, axis=2)

        # Get index of closest segment for each vertex in the chunk
        closest_seg = np.argmin(dist2, axis=1)
        row_indices = np.arange(len(V_chunk))
        t_star = t_val[row_indices, closest_seg]

        # Extract source closest point and map it linearly to the target spine
        P_closest_src = proj[row_indices, closest_seg]
        P_closest_tgt = A_tgt[closest_seg] + t_star[:, None] * D_tgt[closest_seg]

        # Gather frame vectors for source
        T_v_src = T_src[closest_seg]
        N_v_src_curr = N_src[closest_seg]
        next_seg = np.minimum(closest_seg + 1, N_points - 2)
        N_v_src_next = N_src[next_seg]

        N_v_src = (1.0 - t_star[:, None]) * N_v_src_curr + t_star[
            :, None
        ] * N_v_src_next
        N_v_src = N_v_src - np.sum(N_v_src * T_v_src, axis=1, keepdims=True) * T_v_src
        N_v_src_norm = np.linalg.norm(N_v_src, axis=1, keepdims=True)
        N_v_src = N_v_src / np.maximum(N_v_src_norm, 1e-12)
        B_v_src = np.cross(T_v_src, N_v_src)

        # Gather frame vectors for target
        T_v_tgt = T_tgt[closest_seg]
        N_v_tgt_curr = N_tgt[closest_seg]
        N_v_tgt_next = N_tgt[next_seg]

        N_v_tgt = (1.0 - t_star[:, None]) * N_v_tgt_curr + t_star[
            :, None
        ] * N_v_tgt_next
        N_v_tgt = N_v_tgt - np.sum(N_v_tgt * T_v_tgt, axis=1, keepdims=True) * T_v_tgt
        N_v_tgt_norm = np.linalg.norm(N_v_tgt, axis=1, keepdims=True)
        N_v_tgt = N_v_tgt / np.maximum(N_v_tgt_norm, 1e-12)
        B_v_tgt = np.cross(T_v_tgt, N_v_tgt)

        # Relative coordinates in source frame
        d_vec = V_chunk - P_closest_src
        x = np.sum(d_vec * N_v_src, axis=1)
        y = np.sum(d_vec * B_v_src, axis=1)
        z = np.sum(d_vec * T_v_src, axis=1)

        # Reconstruct coordinates in target frame
        V_new[start_idx:end_idx] = (
            P_closest_tgt
            + x[:, None] * N_v_tgt
            + y[:, None] * B_v_tgt
            + z[:, None] * T_v_tgt
        )

    deformed_mesh = mesh.copy()
    deformed_mesh.vertices = V_new
    return deformed_mesh


def deform_mesh_to_spine_bpy(
    mesh: trimesh.Trimesh,
    source_spine: np.ndarray,
    target_spine: np.ndarray,
) -> trimesh.Trimesh:
    """
    Deform a mesh using Blender's Armature modifier (Linear Blend Skinning / LBS) via bpy.
    """
    N_points = len(source_spine)
    if N_points < 2:
        raise ValueError("Spine must have at least 2 points to define segments.")

    # 1. Compute vertex weights using optimized numpy binning
    A_src = source_spine[:-1]
    D_src = source_spine[1:] - source_spine[:-1]
    L2_src = np.sum(D_src**2, axis=1)
    L2_src = np.maximum(L2_src, 1e-12)

    disp = mesh.vertices[:, None, :] - A_src[None, :, :]
    dot = np.sum(disp * D_src[None, :, :], axis=2)
    t_val = np.clip(dot / L2_src[None, :], 0.0, 1.0)
    proj = A_src[None, :, :] + t_val[:, :, None] * D_src[None, :, :]
    dist2 = np.sum((mesh.vertices[:, None, :] - proj) ** 2, axis=2)
    closest_seg = np.argmin(dist2, axis=1)
    t_star = t_val[np.arange(len(mesh.vertices)), closest_seg]

    bins = np.linspace(0.0, 1.0, 21)
    bin_indices = np.digitize(t_star, bins) - 1

    # 2. Setup temporary Blender Armature and Mesh Objects
    arm_data = bpy.data.armatures.new("TempDeformArm")
    arm_obj = bpy.data.objects.new("TempDeformArm", arm_data)
    bpy.context.collection.objects.link(arm_obj)

    mesh_data = bpy.data.meshes.new("TempDeformMesh")
    mesh_obj = bpy.data.objects.new("TempDeformMesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    try:
        # Create Edit Bones matching the source spine
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")

        for i in range(N_points - 1):
            bone = arm_data.edit_bones.new(f"Bone_{i}")
            bone.head = source_spine[i].tolist()
            bone.tail = source_spine[i + 1].tolist()
            if i > 0:
                bone.parent = arm_data.edit_bones[f"Bone_{i - 1}"]
                bone.use_connect = True

        bpy.ops.object.mode_set(mode="OBJECT")

        # Load mesh vertices and faces
        mesh_data.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
        mesh_data.update()

        # Create Vertex Groups and assign weights
        vgs = [
            mesh_obj.vertex_groups.new(name=f"Bone_{i}") for i in range(N_points - 1)
        ]
        for i in range(N_points - 1):
            mask_seg = closest_seg == i
            if not np.any(mask_seg):
                continue
            if i < N_points - 2:
                for b_idx in range(len(bins)):
                    mask_bin = (bin_indices == b_idx) & mask_seg
                    indices = np.where(mask_bin)[0]
                    if len(indices) == 0:
                        continue
                    w_next = float(bins[b_idx])
                    w_curr = 1.0 - w_next
                    if w_curr > 0.0:
                        vgs[i].add(indices.tolist(), w_curr, "REPLACE")
                    if w_next > 0.0:
                        vgs[i + 1].add(indices.tolist(), w_next, "REPLACE")
            else:
                indices = np.where(mask_seg)[0]
                vgs[i].add(indices.tolist(), 1.0, "REPLACE")

        # Add Armature Modifier
        mod = mesh_obj.modifiers.new(name="Arm", type="ARMATURE")
        mod.object = arm_obj
        mod.use_vertex_groups = True

        # Apply target matrices in Pose Mode using relative rotations
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="POSE")

        for i in range(N_points - 1):
            pb = arm_obj.pose.bones[f"Bone_{i}"]

            T_src_vec = mathutils.Vector(source_spine[i + 1] - source_spine[i])
            T_tgt_vec = mathutils.Vector(target_spine[i + 1] - target_spine[i])
            q = T_src_vec.rotation_difference(T_tgt_vec)

            M_bind = pb.bone.matrix_local
            R_bind = M_bind.to_3x3()
            R_pose = q.to_matrix() @ R_bind

            M = mathutils.Matrix.Translation(target_spine[i]) @ R_pose.to_4x4()
            pb.matrix = M

        bpy.ops.object.mode_set(mode="OBJECT")

        # Get deformed vertices from Evaluated Mesh
        dg = bpy.context.evaluated_depsgraph_get()
        eval_mesh_obj = mesh_obj.evaluated_get(dg)
        eval_mesh = eval_mesh_obj.to_mesh()

        new_vertices = np.array([v.co for v in eval_mesh.vertices])
        eval_mesh_obj.to_mesh_clear()

    finally:
        # Clean up temporary datablocks and objects
        if mesh_obj.name in bpy.data.objects:
            bpy.data.objects.remove(mesh_obj, do_unlink=True)
        if arm_obj.name in bpy.data.objects:
            bpy.data.objects.remove(arm_obj, do_unlink=True)
        if mesh_data.name in bpy.data.meshes:
            bpy.data.meshes.remove(mesh_data, do_unlink=True)
        if arm_data.name in bpy.data.armatures:
            bpy.data.armatures.remove(arm_data, do_unlink=True)

    straightened_mesh = mesh.copy()
    straightened_mesh.vertices = new_vertices
    return straightened_mesh


def deform_mesh_to_spine(
    mesh: trimesh.Trimesh,
    source_spine: np.ndarray,
    target_spine: np.ndarray,
    backend: str = "numpy",
    chunk_size: int = 10000,
) -> trimesh.Trimesh:
    """
    Dispatcher to deform a mesh to a target spine using either the 'numpy' or 'bpy' backend.
    """
    if backend == "numpy":
        return deform_mesh_to_spine_numpy(mesh, source_spine, target_spine, chunk_size)
    elif backend == "bpy":
        return deform_mesh_to_spine_bpy(mesh, source_spine, target_spine)
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'numpy' or 'bpy'.")


def straighten(
    mesh: trimesh.Trimesh,
    spine_points: Union[np.ndarray, torch.Tensor, List[Any], Spline],
    num_segments: int | None = None,
    axis: str = "z",
    backend: str = "numpy",
) -> trimesh.Trimesh:
    """
    Straighten a curved mesh by mapping its vertices from the local coordinate frames of a curved
    spine (skeleton path) to a straight line along a specified axis.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input curved mesh.
    spine_points : array-like or Spline
        The control points or evaluated points representing the curved spine.
        - If a Spline object, it is evaluated.
    num_segments : int or None
        The number of interpolation points to use along the spine.
    axis : str
        The axis along which to straighten the mesh ('x', 'y', or 'z').
    backend : str
        The deformation backend to use ('numpy' or 'bpy').

    Returns
    -------
    straightened_mesh : trimesh.Trimesh
        A new mesh representing the straightened geometry.
    """
    if num_segments is None:
        num_segments = 100

    # 1. Resolve spine points to a numpy array of shape (N, 3)
    if isinstance(spine_points, Spline):
        eval_pts = spine_points.evaluate_curve(
            num_points_per_segment=max(5, num_segments // len(spine_points.points) + 1)
        )
        spine_pts_np = np.array([pt.detach().cpu().numpy() for pt in eval_pts])
    elif isinstance(spine_points, torch.Tensor):
        spine_pts_np = spine_points.detach().cpu().numpy()
    elif isinstance(spine_points, np.ndarray):
        spine_pts_np = spine_points
    else:
        resolved = []
        for pt in spine_points:
            if isinstance(pt, torch.Tensor):
                resolved.append(pt.detach().cpu().numpy())
            else:
                resolved.append(np.array(pt))
        spine_pts_np = np.stack(resolved, axis=0)

    if spine_pts_np.ndim != 2 or spine_pts_np.shape[1] != 3:
        raise ValueError(
            f"Spine points must have shape (N, 3), got {spine_pts_np.shape}"
        )

    # Calculate cumulative arc lengths to define the target straight spine
    _, _, _, _, s = build_bishop_frame(spine_pts_np)

    target_spine = np.zeros_like(spine_pts_np)
    if axis == "z":
        target_spine[:, 2] = s
    elif axis == "y":
        target_spine[:, 1] = s
    elif axis == "x":
        target_spine[:, 0] = s
    else:
        raise ValueError(f"Unknown axis: {axis}. Must be 'x', 'y', or 'z'.")

    return deform_mesh_to_spine(mesh, spine_pts_np, target_spine, backend=backend)


def straighten_lateral(
    mesh: trimesh.Trimesh,
    spine_points: Union[np.ndarray, torch.Tensor, List[Any], Spline],
    num_segments: int | None = None,
    straighten_axis: str = "x",
    backend: str = "numpy",
) -> trimesh.Trimesh:
    """
    Straighten a curved mesh along a specific lateral axis (e.g., left-right / X axis),
    while preserving its coordinates and shape along the other axes (e.g., Z / vertical profile).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input curved mesh.
    spine_points : array-like or Spline
        The control points or evaluated points representing the curved spine.
        - If a Spline object, it is evaluated.
    num_segments : int or None
        The number of interpolation points to use along the spine.
    straighten_axis : str
        The lateral coordinate axis to flatten ('x', 'y', or 'z').
    backend : str
        The deformation backend to use ('numpy' or 'bpy').

    Returns
    -------
    straightened_mesh : trimesh.Trimesh
        A new mesh with the lateral curvature removed.
    """
    if num_segments is None:
        num_segments = 100

    # 1. Resolve spine points to a numpy array of shape (N, 3)
    if isinstance(spine_points, Spline):
        eval_pts = spine_points.evaluate_curve(
            num_points_per_segment=max(5, num_segments // len(spine_points.points) + 1)
        )
        spine_pts_np = np.array([pt.detach().cpu().numpy() for pt in eval_pts])
    elif isinstance(spine_points, torch.Tensor):
        spine_pts_np = spine_points.detach().cpu().numpy()
    elif isinstance(spine_points, np.ndarray):
        spine_pts_np = spine_points
    else:
        resolved = []
        for pt in spine_points:
            if isinstance(pt, torch.Tensor):
                resolved.append(pt.detach().cpu().numpy())
            else:
                resolved.append(np.array(pt))
        spine_pts_np = np.stack(resolved, axis=0)

    if spine_pts_np.ndim != 2 or spine_pts_np.shape[1] != 3:
        raise ValueError(
            f"Spine points must have shape (N, 3), got {spine_pts_np.shape}"
        )

    # Create target spine by setting the specified axis to its initial value
    target_spine = spine_pts_np.copy()
    if straighten_axis == "x":
        target_spine[:, 0] = spine_pts_np[0, 0]
    elif straighten_axis == "y":
        target_spine[:, 1] = spine_pts_np[0, 1]
    elif straighten_axis == "z":
        target_spine[:, 2] = spine_pts_np[0, 2]
    else:
        raise ValueError(
            f"Unknown straighten_axis: {straighten_axis}. Must be 'x', 'y', or 'z'."
        )

    return deform_mesh_to_spine(mesh, spine_pts_np, target_spine, backend=backend)
