"""
This module contains functions to output GLB (glTF 2.0 Binary) file format.

It constructs valid glTF 2.0 binary packages with mesh primitives, materials,
skeletal node hierarchies, skinning weights, and animations using pygltflib and trimesh.

NOTE: This should probably be very useful if it is created into a separated library later on.
"""

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pygltflib
import trimesh

from animgen.core.armature import Armature
from animgen.rigging.skinning import compute_auto_skin_weights
from animgen.animation.animator import AnimationClip
from animgen.core.types import Animation
from animgen.utils.math import rotation_matrix_to_quaternion


def _append_binary_buffer(
    gltf: pygltflib.GLTF2,
    data: bytes,
    target: Optional[int] = None,
) -> int:
    """
    Appends raw binary data to the glTF binary blob with 4-byte alignment,
    creates a BufferView, and returns its index.
    """
    bin_data = bytearray(gltf.binary_blob())
    offset = len(bin_data)
    if offset % 4 != 0:
        bin_data.extend(b"\x00" * (4 - (offset % 4)))
        offset = len(bin_data)
    bin_data.extend(data)
    gltf.set_binary_blob(bytes(bin_data))

    bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(data),
            target=target,
        )
    )
    return bv_idx


def _append_accessor(
    gltf: pygltflib.GLTF2,
    buffer_view_idx: int,
    component_type: int,
    count: int,
    type_str: str,
    min_val: Optional[list[float]] = None,
    max_val: Optional[list[float]] = None,
) -> int:
    """
    Appends an Accessor to the glTF structure and returns its index.
    """
    acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=buffer_view_idx,
            byteOffset=0,
            componentType=component_type,
            count=count,
            type=type_str,
            min=min_val,
            max=max_val,
        )
    )
    return acc_idx


def mesh_to_gltf(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> pygltflib.GLTF2:
    """
    Converts a trimesh geometry/scene into a pygltflib GLTF2 structure,
    preserving textures, UV mappings, and PBR materials.

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene
        The input mesh to convert.

    Returns
    -------
    pygltflib.GLTF2
        The parsed GLTF2 structure containing the mesh geometry.
    """
    glb_bytes = mesh.export(file_type="glb")
    return pygltflib.GLTF2().load_from_bytes(glb_bytes)


def add_armature_and_skin(
    gltf: pygltflib.GLTF2,
    mesh: trimesh.Trimesh,
    armature: Armature,
    weights_dict: Optional[dict[str, np.ndarray]] = None,
) -> tuple[pygltflib.GLTF2, dict[str, int]]:
    """
    Attaches an Armature node hierarchy and Skin definition (with JOINTS_0, WEIGHTS_0,
    and inverseBindMatrices) to an existing GLTF2 structure.

    Parameters
    ----------
    gltf : pygltflib.GLTF2
        The GLTF2 object containing the mesh.
    mesh : trimesh.Trimesh
        The reference mesh used for vertex count and skin weight extraction.
    armature : Armature
        The armature hierarchy.
    weights_dict : Optional[dict[str, np.ndarray]]
        Precomputed skinning weights. If None, auto skin weights will be calculated.

    Returns
    -------
    tuple[pygltflib.GLTF2, dict[str, int]]
        The updated GLTF2 structure and a mapping from bone ID to its glTF Node index.
    """
    if weights_dict is None:
        weights_dict = compute_auto_skin_weights(mesh, armature)

    bone_ids = [b.id for b in armature.bones_list]
    num_verts = len(mesh.vertices)

    joints_0 = np.zeros((num_verts, 4), dtype=np.uint16)
    weights_0 = np.zeros((num_verts, 4), dtype=np.float32)

    all_weights = np.column_stack([weights_dict[b_id] for b_id in bone_ids])
    for i in range(num_verts):
        w_row = all_weights[i]
        top_indices = np.argsort(w_row)[::-1][:4]
        top_weights = w_row[top_indices]
        s = top_weights.sum()
        if s > 0:
            top_weights = top_weights / s
        else:
            top_weights[0] = 1.0
        joints_0[i, : len(top_indices)] = top_indices
        weights_0[i, : len(top_weights)] = top_weights

    joints_bv = _append_binary_buffer(
        gltf, joints_0.tobytes(), target=pygltflib.ARRAY_BUFFER
    )
    joints_acc = _append_accessor(
        gltf, joints_bv, pygltflib.UNSIGNED_SHORT, num_verts, pygltflib.VEC4
    )

    weights_bv = _append_binary_buffer(
        gltf, weights_0.tobytes(), target=pygltflib.ARRAY_BUFFER
    )
    weights_acc = _append_accessor(
        gltf, weights_bv, pygltflib.FLOAT, num_verts, pygltflib.VEC4
    )

    inv_bind_matrices = []
    for bone in armature.bones_list:
        M_inv = np.eye(4, dtype=np.float32)
        M_inv[0, 3] = -float(bone.head[0])
        M_inv[1, 3] = -float(bone.head[1])
        M_inv[2, 3] = -float(bone.head[2])
        inv_bind_matrices.append(M_inv.T)  # Column-major for glTF

    inv_bind_bytes = np.array(inv_bind_matrices, dtype=np.float32).tobytes()
    inv_bv = _append_binary_buffer(gltf, inv_bind_bytes)
    inv_acc = _append_accessor(
        gltf, inv_bv, pygltflib.FLOAT, len(armature.bones_list), pygltflib.MAT4
    )

    for prim in gltf.meshes[0].primitives:
        prim.attributes.JOINTS_0 = joints_acc
        prim.attributes.WEIGHTS_0 = weights_acc

    # Build Bone Node Hierarchy
    bone_to_node_idx: dict[str, int] = {}
    joint_indices: list[int] = []

    for bone in armature.bones_list:
        if bone.parent is None:
            trans = [float(bone.head[0]), float(bone.head[1]), float(bone.head[2])]
        else:
            trans = [
                float(bone.head[0] - bone.parent.head[0]),
                float(bone.head[1] - bone.parent.head[1]),
                float(bone.head[2] - bone.parent.head[2]),
            ]
        node = pygltflib.Node(
            name=bone.id,
            translation=trans if trans != [0.0, 0.0, 0.0] else None,
            children=[],
        )
        gltf.nodes.append(node)
        idx = len(gltf.nodes) - 1
        bone_to_node_idx[bone.id] = idx
        joint_indices.append(idx)

    for bone in armature.bones_list:
        if bone.parent is not None:
            p_idx = bone_to_node_idx[bone.parent.id]
            c_idx = bone_to_node_idx[bone.id]
            if gltf.nodes[p_idx].children is None:
                gltf.nodes[p_idx].children = []
            gltf.nodes[p_idx].children.append(c_idx)

    skin_idx = len(gltf.skins)
    gltf.skins.append(
        pygltflib.Skin(
            name="ArmatureSkin",
            inverseBindMatrices=inv_acc,
            joints=joint_indices,
        )
    )

    for n in gltf.nodes:
        if n.mesh is not None:
            n.skin = skin_idx

    root_indices = [
        bone_to_node_idx[r.id]
        for r in armature.disconnected_chain_roots
        if r.id in bone_to_node_idx
    ]
    armature_node = pygltflib.Node(name="Armature", children=root_indices)
    gltf.nodes.append(armature_node)
    arm_node_idx = len(gltf.nodes) - 1

    scene_idx = gltf.scene if gltf.scene is not None else 0
    if gltf.scenes[scene_idx].nodes is None:
        gltf.scenes[scene_idx].nodes = []
    gltf.scenes[scene_idx].nodes.append(arm_node_idx)

    return gltf, bone_to_node_idx


def add_animation(
    gltf: pygltflib.GLTF2,
    animation: Union[AnimationClip, Animation, dict[float, Any]],
    bone_to_node_idx: dict[str, int],
    clip_name: str = "Animation",
    armature: Optional[Armature] = None,
) -> pygltflib.GLTF2:
    """
    Attaches skeletal animation tracks (keyframes / rotation channels)
    to a rigged GLTF2 object.

    Parameters
    ----------
    gltf : pygltflib.GLTF2
        The GLTF2 object containing the armature nodes.
    animation : AnimationClip | Animation | dict
        The animation data or AnimationClip containing time -> frame transformations.
    bone_to_node_idx : dict[str, int]
        Mapping from bone ID to glTF Node index.
    clip_name : str, default='Animation'
        Name of the animation track.
    armature : Optional[Armature], default=None
        Armature structure corresponding to the bones.

    Returns
    -------
    pygltflib.GLTF2
        The updated GLTF2 object with the animation appended.
    """
    if hasattr(animation, "positions"):
        if not animation.positions and hasattr(animation, "generate_animation"):
            animation.generate_animation()
        positions = animation.positions
    else:
        positions = animation

    if not positions:
        return gltf

    timestamps = sorted(positions.keys())
    if len(timestamps) < 2:
        return gltf

    time_array = np.array(timestamps, dtype=np.float32)
    time_bv = _append_binary_buffer(gltf, time_array.tobytes())
    time_acc = _append_accessor(
        gltf,
        time_bv,
        pygltflib.FLOAT,
        len(time_array),
        pygltflib.SCALAR,
        min_val=[float(time_array[0])],
        max_val=[float(time_array[-1])],
    )

    channels: list[pygltflib.AnimationChannel] = []
    samplers: list[pygltflib.AnimationSampler] = []

    bones_list = []
    if armature is not None:
        bones_list = armature.bones_list
    elif hasattr(animation, "armature") and animation.armature is not None:
        bones_list = animation.armature.bones_list

    if bones_list:
        for b_idx, bone in enumerate(bones_list):
            if bone.id not in bone_to_node_idx:
                continue
            node_idx = bone_to_node_idx[bone.id]

            quats = []
            for t in timestamps:
                frame = positions[t]
                if b_idx < len(frame):
                    R = frame[b_idx]
                    q = rotation_matrix_to_quaternion(R)
                    quats.append([float(q[1]), float(q[2]), float(q[3]), float(q[0])])
                else:
                    quats.append([0.0, 0.0, 0.0, 1.0])

            quat_array = np.array(quats, dtype=np.float32)
            quat_bv = _append_binary_buffer(gltf, quat_array.tobytes())
            quat_acc = _append_accessor(
                gltf,
                quat_bv,
                pygltflib.FLOAT,
                len(timestamps),
                pygltflib.VEC4,
            )

            sampler_idx = len(samplers)
            samplers.append(
                pygltflib.AnimationSampler(
                    input=time_acc,
                    interpolation=pygltflib.ANIM_LINEAR,
                    output=quat_acc,
                )
            )
            channels.append(
                pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=node_idx,
                        path="rotation",
                    ),
                )
            )

    gltf_anim = pygltflib.Animation(
        name=clip_name, channels=channels, samplers=samplers
    )
    gltf.animations.append(gltf_anim)
    return gltf


def export_glb(
    mesh: Union[trimesh.Trimesh, trimesh.Scene, Any],
    output_path: str | Path,
    armature: Optional[Armature] = None,
    skin_weights: Optional[dict[str, np.ndarray]] = None,
    animation: Optional[Union[AnimationClip, Animation, dict[float, Any]]] = None,
) -> Path:
    """
    Exports a 3D mesh, optional armature hierarchy with skinning, and optional
    animation tracks to a GLB file.

    Parameters
    ----------
    mesh : trimesh.Trimesh | trimesh.Scene | Any
        The input 3D mesh to export. If an object with a `mesh` attribute is passed,
        its `mesh`, `armature`, and `skin_weights` attributes will be used.
    output_path : str | Path
        Target destination file path (e.g. 'output/rigged_model.glb').
    armature : Optional[Armature]
        Optional Armature hierarchy to bind and export alongside the mesh.
    skin_weights : Optional[dict[str, np.ndarray]]
        Optional precomputed skin weights dictionary mapping bone ID to per-vertex weights.
    animation : Optional[AnimationClip | Animation | dict]
        Optional animation track/clip to embed into the GLB.

    Returns
    -------
    Path
        The resolved Path to the saved GLB file.
    """
    # Extract mesh, armature, and skin_weights from wrapper if passed
    if hasattr(mesh, "mesh"):
        if armature is None and hasattr(mesh, "armature"):
            armature = mesh.armature
        if skin_weights is None and hasattr(mesh, "skin_weights"):
            skin_weights = mesh.skin_weights
        mesh = mesh.mesh

    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise TypeError(
            f"Expected mesh of type trimesh.Trimesh or trimesh.Scene, got {type(mesh)}"
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert mesh to base glTF
    gltf = mesh_to_gltf(mesh)

    # Add Armature & Skin if present
    bone_to_node_idx: dict[str, int] = {}
    if armature is not None and armature.bones_list:
        raw_mesh = mesh.to_mesh() if isinstance(mesh, trimesh.Scene) else mesh
        gltf, bone_to_node_idx = add_armature_and_skin(
            gltf, raw_mesh, armature, weights_dict=skin_weights
        )

    # Add Animation tracks if present
    if animation is not None and bone_to_node_idx:
        gltf = add_animation(gltf, animation, bone_to_node_idx, armature=armature)

    # Save GLB file
    gltf.save(str(out_path))
    return out_path
