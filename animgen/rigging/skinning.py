"""
This module provides skinning utilities using Blender's automatic weighting (ARMATURE_AUTO).

It extracts per-vertex bone influences given a trimesh.Trimesh and an Armature.

TODO: Consider writing seperate implementation to remove bpy dependance.
"""

from typing import Union
import numpy as np
import trimesh

from animgen.core.armature import Armature


def compute_auto_skin_weights(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    armature: Armature,
) -> dict[str, np.ndarray]:
    """
    Computes automatic skinning weights for each vertex and bone using Blender's
    voxel/heat automatic skinning engine (ARMATURE_AUTO).

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
    try:
        import bpy
    except ImportError as e:
        raise ImportError(
            "Blender ('bpy') is required to compute automatic skinning weights. "
            "Please ensure bpy is installed in your python environment."
        ) from e

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh of type trimesh.Trimesh, got {type(mesh)}")

    if armature is None or not armature.bones_list:
        raise ValueError(
            "An Armature with at least one bone must be provided for skinning."
        )

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
