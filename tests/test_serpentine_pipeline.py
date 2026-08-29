from pathlib import Path
import numpy as np
import pytest
import trimesh

from animgen.core.models.model import BaseModelClass
from animgen.core.models.serpentine import SerpentineModels
from animgen.core.armature import Armature
from animgen.animation.animator import Animator


def test_serpentine_pipeline_on_cylinder():
    """
    Test SerpentineModels end-to-end on a segmented cylinder mesh.
    """
    # Create a curved tube / cylinder mesh along the X-axis
    cylinder = trimesh.creation.cylinder(radius=0.2, height=4.0, sections=16)
    # Rotate cylinder to lie along X-axis
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    cylinder.apply_transform(rot)

    model = BaseModelClass(cylinder)
    assert model.mesh is not None

    pipeline = SerpentineModels(
        model=model,
        num_bones=10,
        frame_rate=10.0,
    )

    # Execute end-to-end pipeline
    processed_model = pipeline.process()

    assert processed_model.mesh is not None
    assert isinstance(processed_model.armature, Armature)
    assert len(processed_model.armature.bones_list) == 10

    assert isinstance(processed_model.animator, Animator)
    assert "slow" in processed_model.animator.animations
    assert "fast" in processed_model.animator.animations

    assert processed_model.skin_weights is not None
    assert len(processed_model.skin_weights) == 10

    # Test that animator can evaluate / bake frames for both slow and fast
    baked = processed_model.animator.bake(mesh=processed_model.mesh)
    assert "slow" in baked
    assert "fast" in baked
    assert len(baked["slow"]) == 30  # 30 frames for 3.0s at 10 FPS
    assert len(baked["fast"]) == 12  # 12 frames for 1.2s at 10 FPS


@pytest.mark.slow
def test_serpentine_rigging_on_paint_mesh():
    """
    1. Test SerpentineModels autorig() stage on textured paint_mesh_Sea_Snake.
    Exports the base rigged model to tests/artifacts/test_serpentine_rigged_snake.glb.
    """
    snake_path = Path("generated_data/models/paint_mesh_Sea_Snake.glb")
    if not snake_path.exists():
        snake_path = Path(
            "generated_data/models/models_backup_3/paint_mesh_Sea_Snake.glb"
        )
    if not snake_path.exists():
        pytest.skip(f"Textured sea snake mesh not found at {snake_path}")

    model = BaseModelClass(snake_path)
    assert model.mesh is not None
    orig_num_verts = len(model.mesh.vertices)
    orig_num_faces = len(model.mesh.faces)

    pipeline = SerpentineModels(model=model, num_bones=20)

    # 1. Execute autorig()
    rigged_model = pipeline.autorig()

    assert rigged_model.mesh is not None
    assert len(rigged_model.mesh.vertices) == orig_num_verts
    assert len(rigged_model.mesh.faces) == orig_num_faces

    assert isinstance(rigged_model.armature, Armature)
    assert len(rigged_model.armature.bones_list) == 20

    assert rigged_model.skin_weights is not None
    assert len(rigged_model.skin_weights) == 20

    # Verify bone chain connectivity
    armature = rigged_model.armature
    for i in range(len(armature.bones_list) - 1):
        parent_bone = armature.bones_list[i]
        child_bone = armature.bones_list[i + 1]
        assert child_bone.parent == parent_bone
        np.testing.assert_allclose(child_bone.head, parent_bone.tail, atol=1e-6)

    # Export Base Rigged GLB
    out_dir = Path("tests/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_rigged = out_dir / "test_serpentine_rigged_snake.glb"

    exported_path = rigged_model.export(out_path_rigged)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0
    print(
        f"\n[1. Base Rigged GLB Exported]: {exported_path.resolve()} ({exported_path.stat().st_size} bytes)"
    )


@pytest.mark.slow
def test_serpentine_animation_on_paint_mesh():
    """
    Test SerpentineModels process() end-to-end on textured paint_mesh_Sea_Snake.
    Exports the skeletal animated GLB with embedded glTF animation tracks.
    """
    snake_path = Path("generated_data/models/paint_mesh_Sea_Snake.glb")
    if not snake_path.exists():
        snake_path = Path(
            "generated_data/models/models_backup_3/paint_mesh_Sea_Snake.glb"
        )
    if not snake_path.exists():
        pytest.skip(f"Textured sea snake mesh not found at {snake_path}")

    model = BaseModelClass(snake_path)
    assert model.mesh is not None

    pipeline = SerpentineModels(
        model=model,
        num_bones=20,
        frame_rate=30.0,
    )

    # Execute full end-to-end pipeline
    animated_model = pipeline.process()

    assert animated_model.mesh is not None
    assert isinstance(animated_model.armature, Armature)
    assert len(animated_model.armature.bones_list) == 20

    # Ensure mesh bounds and armature heads/tails align along X
    mesh_x_min, mesh_x_max = (
        animated_model.mesh.bounds[0, 0],
        animated_model.mesh.bounds[1, 0],
    )
    arm_x_min, arm_x_max = (
        animated_model.armature.bones_list[0].head[0],
        animated_model.armature.bones_list[-1].tail[0],
    )
    assert abs(mesh_x_min - arm_x_min) < 0.1
    assert abs(mesh_x_max - arm_x_max) < 0.1

    assert isinstance(animated_model.animator, Animator)
    assert "slow" in animated_model.animator.animations
    assert "fast" in animated_model.animator.animations

    assert animated_model.skin_weights is not None
    assert len(animated_model.skin_weights) == 20

    out_dir = Path("tests/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export Skeletal Animated GLB
    out_path_animated = out_dir / "test_serpentine_animated_snake.glb"
    exported_anim = animated_model.export(
        out_path_animated, animation=animated_model.animator
    )
    assert exported_anim.exists()
    assert exported_anim.stat().st_size > 0
    print(
        f"\n[Skeletal Animated GLB Exported]: {exported_anim.resolve()} ({exported_anim.stat().st_size} bytes)"
    )
