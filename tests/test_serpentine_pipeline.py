from pathlib import Path
import numpy as np
import pytest
import trimesh

from animgen.core.models.model import BaseModelClass
from animgen.core.models.serpentine import SerpentineModels
from animgen.core.armature import Armature
from animgen.animation.animator import Animator


def _get_snake_mesh_path(filename: str) -> Path | None:
    for candidate in [
        Path("generated_data/models") / filename,
        Path("generated_data/models/models_backup_3") / filename,
    ]:
        if candidate.exists():
            return candidate
    return None


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


def test_serpentine_canonicalize_head_tail_detection_on_tapered_cylinder():
    """
    Tests that SerpentineModels.canonicalize() automatically identifies head vs tail
    and straightens the model so head is at root (x=0) and tail is at x=L even if input is flipped.
    """
    # Create tapered cylinder: thick at +X, thin at -X (flipped input)
    cylinder = trimesh.creation.cylinder(radius=0.3, height=3.0, sections=16)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    cylinder.apply_transform(rot)

    # Taper so +X is thick (radius 0.4) and -X is narrow (radius 0.05)
    v = cylinder.vertices.copy()
    scale_factor = 0.5 + 0.4 * (v[:, 0] / 1.5)
    v[:, 1] *= scale_factor
    v[:, 2] *= scale_factor
    cylinder.vertices = v

    model = BaseModelClass(cylinder)
    pipe = SerpentineModels(model, num_bones=10)

    # Run canonicalize
    straightened = pipe.canonicalize(pipe.segment())

    # After canonicalize, thicker head should be at root (x=0)
    sv = straightened.vertices
    xmin, xmax = sv[:, 0].min(), sv[:, 0].max()
    span = xmax - xmin
    mask_root = sv[:, 0] <= xmin + 0.10 * span
    mask_tip = sv[:, 0] >= xmax - 0.10 * span

    r_root = np.linalg.norm(sv[mask_root, 1:], axis=1).mean()
    r_tip = np.linalg.norm(sv[mask_tip, 1:], axis=1).mean()

    assert r_root > r_tip, "Thick head was not placed at root x=0"


@pytest.mark.slow
@pytest.mark.parametrize(
    "mesh_name", ["paint_mesh_Sea_Snake.glb", "dec_mesh_Sea_Snake.glb"]
)
def test_serpentine_rigging_on_mesh(mesh_name: str):
    """
    Test SerpentineModels autorig() stage on both paint and dec snake meshes.
    """
    snake_path = _get_snake_mesh_path(mesh_name)
    if snake_path is None:
        pytest.skip(f"Snake mesh not found: {mesh_name}")

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
    out_path_rigged = out_dir / f"test_serpentine_rigged_{mesh_name}"

    exported_path = rigged_model.export(out_path_rigged)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0
    print(
        f"\n[Rigged GLB Exported]: {exported_path.resolve()} ({exported_path.stat().st_size} bytes)"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "mesh_name", ["paint_mesh_Sea_Snake.glb", "dec_mesh_Sea_Snake.glb"]
)
def test_serpentine_animation_on_mesh(mesh_name: str):
    """
    Test SerpentineModels process() end-to-end on both paint and dec snake meshes.
    Exports the skeletal animated GLB with embedded glTF animation tracks.
    """
    snake_path = _get_snake_mesh_path(mesh_name)
    if snake_path is None:
        pytest.skip(f"Snake mesh not found: {mesh_name}")

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
    out_path_animated = out_dir / f"test_serpentine_animated_{mesh_name}"
    exported_anim = animated_model.export(
        out_path_animated, animation=animated_model.animator
    )
    assert exported_anim.exists()
    assert exported_anim.stat().st_size > 0
    print(
        f"\n[Animated GLB Exported]: {exported_anim.resolve()} ({exported_anim.stat().st_size} bytes)"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "mesh_name", ["paint_mesh_Sea_Snake.glb", "dec_mesh_Sea_Snake.glb"]
)
def test_serpentine_canonicalize_head_tail_detection_on_snake_mesh(mesh_name: str):
    """
    Tests that SerpentineModels.canonicalize() automatically identifies head vs tail
    and straightens the model so the thicker head is at root (x=0) and tail is at x=L
    even when deliberately inverted 180 degrees.
    """
    snake_path = _get_snake_mesh_path(mesh_name)
    if snake_path is None:
        pytest.skip(f"Snake mesh not found: {mesh_name}")

    mesh = trimesh.load(snake_path, force="mesh")
    # Invert 180 degrees around Y (Head at +X, Tail at -X)
    rot_180 = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    mesh.apply_transform(rot_180)

    pipe = SerpentineModels(BaseModelClass(mesh), num_bones=20)
    straightened = pipe.canonicalize(pipe.segment())

    # Verify head is at root (x=0) and tail is at tip (x=L)
    sv = straightened.vertices
    xmin, xmax = sv[:, 0].min(), sv[:, 0].max()
    span = xmax - xmin
    mask_root = sv[:, 0] <= xmin + 0.10 * span
    mask_tip = sv[:, 0] >= xmax - 0.10 * span

    r_root = float(np.linalg.norm(sv[mask_root, 1:], axis=1).mean())
    r_tip = float(np.linalg.norm(sv[mask_tip, 1:], axis=1).mean())

    assert r_root > r_tip, f"Thicker head was not placed at root x=0 on {mesh_name}"


def test_serpentine_reverse_parameter():
    """
    Tests that reverse=True in SerpentineModels inverts the spine canonicalization direction.
    - reverse=False: Head (thicker end) is at x=0 (root), Tail is at +X
    - reverse=True: Tail (tapered end) is at x=0 (root), Head is at +X
    """
    snake_path = _get_snake_mesh_path("paint_mesh_Sea_Snake.glb")
    if snake_path is None:
        pytest.skip("Snake mesh not found")

    mesh = trimesh.load(snake_path, force="mesh")
    # 1. Normal canonicalization (reverse=False, default): root at head
    pipe_normal = SerpentineModels(BaseModelClass(mesh.copy()), reverse=False)
    segs_n = pipe_normal.segment()
    straightened_n = pipe_normal.canonicalize(segs_n)
    v_n = straightened_n.vertices
    xmin_n, xmax_n = v_n[:, 0].min(), v_n[:, 0].max()
    span_n = xmax_n - xmin_n
    head_v_n = v_n[v_n[:, 0] <= xmin_n + 0.10 * span_n]
    tail_v_n = v_n[v_n[:, 0] >= xmax_n - 0.10 * span_n]
    r_head_n = np.linalg.norm(head_v_n[:, 1:], axis=1).mean()
    r_tail_n = np.linalg.norm(tail_v_n[:, 1:], axis=1).mean()
    assert r_head_n > r_tail_n, (
        "Default canonicalization did not place thicker head at x=0"
    )

    # 2. Reversed canonicalization (reverse=True): root at tail
    pipe_rev = SerpentineModels(BaseModelClass(mesh.copy()), reverse=True)
    segs_r = pipe_rev.segment()
    straightened_r = pipe_rev.canonicalize(segs_r)
    v_r = straightened_r.vertices
    xmin_r, xmax_r = v_r[:, 0].min(), v_r[:, 0].max()
    span_r = xmax_r - xmin_r
    tip0_v_r = v_r[v_r[:, 0] <= xmin_r + 0.10 * span_r]
    tip1_v_r = v_r[v_r[:, 0] >= xmax_r - 0.10 * span_r]
    r_tip0_r = np.linalg.norm(tip0_v_r[:, 1:], axis=1).mean()
    r_tip1_r = np.linalg.norm(tip1_v_r[:, 1:], axis=1).mean()
    assert r_tip1_r > r_tip0_r, (
        "Reversed canonicalization did not place thicker head at +X"
    )
