from pathlib import Path
import numpy as np
import pygltflib
import pytest
import trimesh

from animgen.core.armature import Armature, Bone
from animgen.core.models.model import BaseModelClass
from animgen.io.glb_output import (
    add_animation,
    add_armature_and_skin,
    export_glb,
    mesh_to_gltf,
)
from animgen.rigging.skinning import (
    compute_auto_skin_weights,
    get_skinning_weight_matrix,
)


def test_mesh_to_gltf():
    box = trimesh.creation.box()
    gltf = mesh_to_gltf(box)
    assert len(gltf.meshes) == 1
    assert len(gltf.nodes) >= 1
    assert gltf.meshes[0].primitives[0].attributes.POSITION is not None


def test_add_armature_and_skin():
    box = trimesh.creation.box()
    gltf = mesh_to_gltf(box)

    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5))
    armature = Armature(root)
    b1 = armature.add_connected_bone(root, tail=(0.0, 0.0, 1.0))

    gltf, bone_to_node = add_armature_and_skin(gltf, box, armature)

    assert "root_bone" in bone_to_node
    assert b1.id in bone_to_node
    assert len(gltf.skins) == 1
    mesh_nodes = [n for n in gltf.nodes if n.mesh is not None]
    assert len(mesh_nodes) >= 1
    assert mesh_nodes[0].skin == 0
    assert gltf.meshes[0].primitives[0].attributes.JOINTS_0 is not None
    assert gltf.meshes[0].primitives[0].attributes.WEIGHTS_0 is not None


def test_add_animation_to_gltf():
    box = trimesh.creation.box()
    gltf = mesh_to_gltf(box)

    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5))
    armature = Armature(root)
    gltf, bone_to_node = add_armature_and_skin(gltf, box, armature)

    dummy_anim = {
        0.0: [np.eye(3, dtype=np.float32)],
        0.5: [np.eye(3, dtype=np.float32)],
        1.0: [np.eye(3, dtype=np.float32)],
    }
    gltf = add_animation(gltf, dummy_anim, bone_to_node, clip_name="TestWave")
    assert len(gltf.animations) == 1
    assert gltf.animations[0].name == "TestWave"


def test_export_glb_mesh_only(tmp_path: Path):
    box = trimesh.creation.box()
    out_file = tmp_path / "test_box.glb"

    res = export_glb(box, out_file)
    assert res.exists()

    loaded = trimesh.load(str(out_file))
    if isinstance(loaded, trimesh.Scene):
        loaded_mesh = loaded.to_mesh()
    else:
        loaded_mesh = loaded

    assert len(loaded_mesh.vertices) == len(box.vertices)
    assert len(loaded_mesh.faces) == len(box.faces)


def test_export_glb_with_materials(tmp_path: Path):
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(
        uv=np.random.rand(len(box.vertices), 2),
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[1.0, 0.5, 0.2, 1.0],
            metallicFactor=0.8,
            roughnessFactor=0.2,
        ),
    )
    out_file = tmp_path / "test_material_box.glb"

    res = export_glb(box, out_file)
    assert res.exists()

    gltf = pygltflib.GLTF2().load(str(out_file))
    assert len(gltf.materials) >= 1
    pbr = gltf.materials[0].pbrMetallicRoughness
    assert pbr is not None
    assert np.allclose(pbr.baseColorFactor, [1.0, 0.5, 0.2, 1.0], atol=1e-2)
    assert np.isclose(pbr.metallicFactor, 0.8, atol=1e-2)
    assert np.isclose(pbr.roughnessFactor, 0.2, atol=1e-2)


def test_export_glb_with_armature_and_skinning(tmp_path: Path):
    box = trimesh.creation.box()
    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5))
    armature = Armature(root)
    armature.add_connected_bone(root, tail=(0.0, 0.0, 1.0))

    out_file = tmp_path / "test_armature_box.glb"
    res = export_glb(box, out_file, armature=armature)
    assert res.exists()

    gltf = pygltflib.GLTF2().load(str(out_file))
    assert len(gltf.skins) >= 1
    assert len(gltf.skins[0].joints) >= 2


def test_compute_auto_skin_weights():
    box = trimesh.creation.box()
    root = Bone(id="root_bone", head=(0.0, 0.0, -0.5), tail=(0.0, 0.0, 0.0))
    armature = Armature(root)
    b1 = armature.add_connected_bone(root, tail=(0.0, 0.0, 0.5))

    weights = compute_auto_skin_weights(box, armature)
    assert "root_bone" in weights
    assert b1.id in weights
    assert len(weights["root_bone"]) == len(box.vertices)
    assert len(weights[b1.id]) == len(box.vertices)

    matrix, bone_ids = get_skinning_weight_matrix(box, armature)
    assert matrix.shape == (len(box.vertices), 2)
    assert bone_ids == ["root_bone", b1.id]


def test_export_glb_via_base_model_class(tmp_path: Path):
    box = trimesh.creation.box()
    model = BaseModelClass(box)
    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5))
    model.armature = Armature(root)

    # Test compute_skin_weights method on BaseModelClass
    weights = model.compute_skin_weights()
    assert "root_bone" in weights
    assert model.skin_weights is not None

    out_file = tmp_path / "test_model_class.glb"
    res = model.export(out_file)
    assert res.exists()

    gltf = pygltflib.GLTF2().load(str(out_file))
    assert len(gltf.skins) >= 1


def test_export_glb_with_precomputed_skin_weights(tmp_path: Path):
    box = trimesh.creation.box()
    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5))
    armature = Armature(root)

    custom_weights = {"root_bone": np.ones(len(box.vertices), dtype=np.float32)}
    out_file = tmp_path / "test_custom_weights.glb"
    res = export_glb(box, out_file, armature=armature, skin_weights=custom_weights)
    assert res.exists()

    gltf = pygltflib.GLTF2().load(str(out_file))
    assert len(gltf.skins) >= 1


def test_export_glb_invalid_asset_type():
    with pytest.raises(TypeError):
        export_glb("invalid_mesh", "out.glb")


def test_export_glb_with_multiple_animations(tmp_path: Path):
    from animgen.animation.clip import AnimationClip
    from animgen.animation.animator import Animator

    box = trimesh.creation.box()
    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5))
    armature = Armature(root)

    clip1 = AnimationClip(name="Clip1", duration=1.0, armature=armature)
    clip1.positions = {
        0.0: [np.eye(3, dtype=np.float32)],
        1.0: [np.eye(3, dtype=np.float32)],
    }
    clip2 = AnimationClip(name="Clip2", duration=2.0, armature=armature)
    clip2.positions = {
        0.0: [np.eye(3, dtype=np.float32)],
        2.0: [np.eye(3, dtype=np.float32)],
    }

    # Test list of clips
    out_file1 = tmp_path / "multi_clips.glb"
    res1 = export_glb(box, out_file1, armature=armature, animation=[clip1, clip2])
    assert res1.exists()
    gltf1 = pygltflib.GLTF2().load(str(out_file1))
    assert len(gltf1.animations) == 2
    assert gltf1.animations[0].name == "Clip1"
    assert gltf1.animations[1].name == "Clip2"

    # Test Animator instance
    animator = Animator(armature=armature)
    animator.add_animation_clip(clip1)
    animator.add_animation_clip(clip2)
    out_file2 = tmp_path / "animator.glb"
    res2 = export_glb(box, out_file2, armature=armature, animation=animator)
    assert res2.exists()
    gltf2 = pygltflib.GLTF2().load(str(out_file2))
    assert len(gltf2.animations) == 2
    assert gltf2.animations[0].name == "Clip1"
    assert gltf2.animations[1].name == "Clip2"
