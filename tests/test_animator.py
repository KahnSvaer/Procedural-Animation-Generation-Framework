from pathlib import Path
import numpy as np
import pytest
import torch
import trimesh

from animgen.animation.animator import AnimationClip, Animator
from animgen.animation.deformation import (
    apply_dual_quaternion_skinning_deformation,
    apply_linear_blend_skinning_deformation,
    apply_mesh_deformation,
)
from animgen.animation.kinematics import (
    compute_forward_kinematics,
    successive_rotations,
)
from animgen.animation.wave import chain_wave_generator
from animgen.core.armature import Armature, Bone
from animgen.io.glb_output import export_glb
from animgen.rigging.skinning import compute_auto_skin_weights
from animgen.utils.math import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    slerp_rotation_matrix,
)


def test_quaternion_roundtrip():
    angle = np.pi / 3.0
    c, s = np.cos(angle), np.sin(angle)
    R_expected = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    q = rotation_matrix_to_quaternion(R_expected)
    R_out = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(R_out, R_expected, atol=1e-6)

    # Test direct slerp utility
    q2 = rotation_matrix_to_quaternion(np.eye(3))
    q_slerp = quaternion_slerp(q, q2, alpha=0.5)
    assert len(q_slerp) == 4

    # Test conjugate and multiply
    q_conj = quaternion_conjugate(q)
    q_identity = quaternion_multiply(q, q_conj)
    np.testing.assert_allclose(q_identity, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_slerp_rotation_matrix():
    R1 = np.eye(3)
    # 90 degrees around Z
    R2 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Alpha = 0.0 -> R1
    r_start = slerp_rotation_matrix(R1, R2, alpha=0.0).numpy()
    np.testing.assert_allclose(r_start, R1, atol=1e-6)

    # Alpha = 1.0 -> R2
    r_end = slerp_rotation_matrix(R1, R2, alpha=1.0).numpy()
    np.testing.assert_allclose(r_end, R2, atol=1e-6)

    # Alpha = 0.5 -> 45 degrees around Z
    r_mid = slerp_rotation_matrix(R1, R2, alpha=0.5).numpy()
    c, s = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
    expected_mid = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(r_mid, expected_mid, atol=1e-6)


def test_compute_forward_kinematics():
    # Setup 2-bone connected chain along Z
    root = Bone(id="root", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 1.0))
    armature = Armature(root)
    b1 = armature.add_connected_bone(root, tail=(0.0, 0.0, 2.0))
    # Add unconnected child to root
    b2 = armature.add_unconnected_bone(root, head=(1.0, 0.0, 1.0), tail=(2.0, 0.0, 1.0))

    # Frame with identity
    frame_identity = [torch.eye(3), torch.eye(3), torch.eye(3)]
    _, positions = compute_forward_kinematics(armature, frame_identity)

    np.testing.assert_allclose(positions["root"][0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(positions["root"][1], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(positions[b1.id][0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(positions[b1.id][1], [0.0, 0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(positions[b2.id][0], [1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(positions[b2.id][1], [2.0, 0.0, 1.0], atol=1e-6)

    # Frame where root is rotated 90 degrees around X
    R_x90 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    frame_rot = [R_x90, torch.eye(3), torch.eye(3)]
    _, positions = compute_forward_kinematics(armature, frame_rot)

    # Root tail should rotate from (0, 0, 1) to (0, -1, 0)
    np.testing.assert_allclose(positions["root"][1], [0.0, -1.0, 0.0], atol=1e-6)
    # b1 is connected child of root: head at root's tail (0, -1, 0), tail at (0, -2, 0)
    np.testing.assert_allclose(positions[b1.id][0], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(positions[b1.id][1], [0.0, -2.0, 0.0], atol=1e-6)
    # b2 is unconnected child: rest head is (1, 0, 1) -> offset from root head (0,0,0) is (1, 0, 1) -> rotated to (1, -1, 0)
    np.testing.assert_allclose(positions[b2.id][0], [1.0, -1.0, 0.0], atol=1e-6)


def test_successive_rotations():
    # Straight chain along X
    src = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # Target bent 90 degrees at joint 0
    tgt = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rotations = successive_rotations(src, tgt)
    assert len(rotations) == 2
    # Joint 0 rotates 90 degrees around Z: [1,0,0] -> [0,1,0]
    # Joint 1 needs no additional relative rotation because parent already aligned the second segment
    np.testing.assert_allclose(rotations[1].numpy(), np.eye(3), atol=1e-5)


def test_dqs_and_lbs_mesh_deformation():
    box = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    root = Bone(id="root", head=(0.0, 0.0, -1.0), tail=(0.0, 0.0, 0.0))
    armature = Armature(root)
    b1 = armature.add_connected_bone(root, tail=(0.0, 0.0, 1.0))

    weights = compute_auto_skin_weights(box, armature)
    assert "root" in weights
    assert b1.id in weights

    # Apply identity deformation with both DQS and LBS
    global_rots = {"root": np.eye(3), b1.id: np.eye(3)}
    global_heads = {
        "root": np.array([0.0, 0.0, -1.0]),
        b1.id: np.array([0.0, 0.0, 0.0]),
    }

    deformed_dqs = apply_dual_quaternion_skinning_deformation(
        box, armature, global_rots, global_heads, weights
    )
    np.testing.assert_allclose(deformed_dqs.vertices, box.vertices, atol=1e-6)

    deformed_lbs = apply_linear_blend_skinning_deformation(
        box, armature, global_rots, global_heads, weights
    )
    np.testing.assert_allclose(deformed_lbs.vertices, box.vertices, atol=1e-6)

    # General helper dispatcher
    deformed_gen = apply_mesh_deformation(
        box, armature, global_rots, global_heads, weights, method="dqs"
    )
    np.testing.assert_allclose(deformed_gen.vertices, box.vertices, atol=1e-6)


def test_animator_skin_weights_property_and_validation():
    box = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    root = Bone(id="root", head=(0.0, 0.0, -1.0), tail=(0.0, 0.0, 0.0))
    armature = Armature(root)
    _ = armature.add_connected_bone(root, tail=(0.0, 0.0, 1.0))

    animator = Animator(armature=armature)
    clip = AnimationClip(name="TestClip", duration=1.0, armature=armature)
    clip.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        wave_amplitude=0.1,
        wave_duration=1.0,
        frame_rate=5.0,
    )
    animator.add_animation_clip(clip)

    assert animator.skin_weights is None

    # Calling bake without skin weights must raise ValueError
    with pytest.raises(ValueError, match="skin_weights must be provided"):
        animator.bake(mesh=box)

    # Compute weights externally from rigging and assign to animator
    computed_weights = compute_auto_skin_weights(box, armature)
    animator.skin_weights = computed_weights
    assert animator.skin_weights == computed_weights

    # Baking now succeeds with default DQS
    baked = animator.bake(mesh=box)
    assert "TestClip" in baked
    assert len(baked["TestClip"]) == 5

    # Baking succeeds with LBS
    baked_lbs = animator.bake(mesh=box, method="lbs")
    assert len(baked_lbs["TestClip"]) == 5


def test_animation_clip_multi_track_aggregation():
    # Armature with 3 bones: 2 on spine, 1 side fin
    root = Bone(id="spine_0", head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(root)
    _ = armature.add_connected_bone(root, tail=(2.0, 0.0, 0.0))
    _ = armature.add_unconnected_bone(root, head=(0.5, 0.5, 0.0), tail=(0.5, 1.5, 0.0))

    clip = AnimationClip(name="SwimClip", duration=2.0, armature=armature)

    # Add wave to spine bones [0, 1]
    clip.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        start_offset=0.0,
        wave_amplitude=0.2,
        wave_duration=2.0,
        frame_rate=10.0,
        wave="travelling",
    )

    # Add wave to fin bone [2] with offset 0.5s
    clip.add_animation_movements(
        chain_wave_generator,
        list_bones=[2],
        start_offset=0.5,
        wave_amplitude=0.3,
        wave_duration=2.0,
        frame_rate=10.0,
        wave="standing",
    )

    anim = clip.generate_animation()

    assert len(anim) > 0
    assert len(clip.positions) == len(anim)

    # Verify each frame has 3 rotation matrices (one per bone)
    for _, frame in anim.items():
        assert len(frame) == 3
        for R in frame:
            assert R.shape == (3, 3)
            np.testing.assert_allclose((R.T @ R).numpy(), np.eye(3), atol=1e-5)


def test_animation_clip_loopable_check():
    b0 = Bone(id="b0", head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(b0)
    _ = armature.add_connected_bone(b0, tail=(2.0, 0.0, 0.0))

    clip = AnimationClip(name="LoopingWave", duration=2.0, armature=armature)
    clip.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        start_offset=0.0,
        wave_amplitude=0.2,
        wave_duration=2.0,
        frame_rate=10.0,
        wave="standing",
    )
    clip.generate_animation()

    # Standing wave over 1 full period is periodic
    assert clip.check_loopable() is True
    assert clip.loopable is True


def test_animation_clip_evaluate_and_slerp():
    b0 = Bone(id="b0", head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(b0)

    clip = AnimationClip(name="TestEval", duration=1.0, armature=armature)

    # Manually populate keyframes at 0.0 and 1.0
    R0 = torch.eye(3)
    R1 = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    clip.positions = {
        0.0: [R0],
        1.0: [R1],
    }

    # Evaluate exact endpoints
    frame_0 = clip.evaluate(0.0)
    np.testing.assert_allclose(frame_0[0].numpy(), R0.numpy(), atol=1e-6)

    frame_1 = clip.evaluate(1.0)
    np.testing.assert_allclose(frame_1[0].numpy(), R1.numpy(), atol=1e-6)

    # Evaluate midpoint (0.5s) -> 45 degrees
    frame_mid = clip.evaluate(0.5)
    c, s = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
    expected_mid = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(frame_mid[0].numpy(), expected_mid, atol=1e-6)


def test_animation_clip_bake_mesh():
    box = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    root = Bone(id="root", head=(0.0, 0.0, -1.0), tail=(0.0, 0.0, 0.0))
    armature = Armature(root)
    _ = armature.add_connected_bone(root, tail=(0.0, 0.0, 1.0))

    clip = AnimationClip(name="BakeClip", duration=1.0, armature=armature)
    clip.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        start_offset=0.0,
        wave_amplitude=0.2,
        wave_duration=1.0,
        frame_rate=5.0,
        wave="travelling",
    )

    # Calling bake without skin weights must raise ValueError
    with pytest.raises(ValueError, match="skin_weights must be provided"):
        clip.bake(mesh=box)

    # Pass weights explicitly and test default DQS
    weights = compute_auto_skin_weights(box, armature)
    baked = clip.bake(mesh=box, skin_weights=weights, method="dqs")
    assert len(baked) == 5
    for _, baked_mesh in baked.items():
        assert isinstance(baked_mesh, trimesh.Trimesh)
        assert len(baked_mesh.vertices) == len(box.vertices)
        assert len(baked_mesh.faces) == len(box.faces)

    # Test LBS
    baked_lbs = clip.bake(mesh=box, skin_weights=weights, method="lbs")
    assert len(baked_lbs) == 5


def test_animator_management_and_export(tmp_path: Path):
    box = trimesh.creation.box()
    root = Bone(id="root_bone", head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 1.0))
    armature = Armature(root)
    _ = armature.add_connected_bone(root, tail=(0.0, 0.0, 2.0))

    weights = compute_auto_skin_weights(box, armature)
    animator = Animator(armature=armature, skin_weights=weights)
    clip1 = AnimationClip(name="Wave1", duration=1.0, armature=armature)
    clip1.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        wave_amplitude=0.1,
        wave_duration=1.0,
        frame_rate=5.0,
    )
    clip2 = AnimationClip(name="Wave2", duration=2.0, armature=armature)
    clip2.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        wave_amplitude=0.2,
        wave_duration=2.0,
        frame_rate=5.0,
    )

    animator.add_animation_clip(clip1)
    animator.add_animation_clip(clip2)

    assert len(animator) == 2
    assert "Wave1" in animator
    assert "Wave2" in animator
    assert "Wave3" not in animator
    assert animator.get_animation_clip("Wave1") == clip1
    assert animator.get("Wave1") == clip1
    assert animator.get("NonExistent", None) is None
    assert animator["Wave1"] == clip1
    assert set(animator.keys()) == {"Wave1", "Wave2"}
    assert list(animator.values()) == [clip1, clip2]
    assert dict(animator.items()) == {"Wave1": clip1, "Wave2": clip2}
    assert list(animator) == ["Wave1", "Wave2"]
    assert animator.animations == {"Wave1": clip1, "Wave2": clip2}

    # Generate all
    all_anims = animator.generate_all_animations()
    assert "Wave1" in all_anims
    assert "Wave2" in all_anims

    # Evaluate through animator
    f = animator.evaluate("Wave1", 0.2)
    assert len(f) == 2

    # Bake through animator with DQS
    baked_all = animator.bake(mesh=box, method="dqs")
    assert "Wave1" in baked_all
    assert "Wave2" in baked_all
    assert len(baked_all["Wave1"]) == 5

    # Export to GLB
    out_glb = tmp_path / "test_animated_export.glb"
    res = export_glb(mesh=box, output_path=out_glb, armature=armature, animation=clip1)
    assert res.exists()


def test_animation_clip_steer_rotation():
    root = Bone(id="b0", head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(root)
    _ = armature.add_connected_bone(root, tail=(2.0, 0.0, 0.0))

    # Create 90-degree yaw rotation matrix around Z
    steer_z_90 = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    clip_normal = AnimationClip(name="NormalWave", duration=1.0, armature=armature)
    clip_normal.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        wave_amplitude=0.1,
        wave_duration=1.0,
        frame_rate=5.0,
    )
    anim_normal = clip_normal.generate_animation()

    clip_steered = AnimationClip(name="SteeredWave", duration=1.0, armature=armature)
    clip_steered.add_animation_movements(
        chain_wave_generator,
        list_bones=[0, 1],
        wave_amplitude=0.1,
        wave_duration=1.0,
        frame_rate=5.0,
        steer_rotation=steer_z_90,
    )
    anim_steered = clip_steered.generate_animation()

    for t in anim_normal.keys():
        for b_idx in range(2):
            r_normal = anim_normal[t][b_idx]
            r_steered = anim_steered[t][b_idx]
            steer_typed = steer_z_90.to(r_normal.dtype)
            expected = steer_typed @ r_normal @ steer_typed.T
            np.testing.assert_allclose(
                r_steered.numpy(),
                expected.numpy(),
                atol=1e-5,
            )


def test_animation_clip_timeline_offset_and_wave_phi_t():
    """
    Test the distinction between:
    1. timeline_offset: delays/shifts when the movement begins on the clip timeline.
    2. phi_t: temporal phase offset in the wave equation (shifts starting wave phase at t=0).
    """
    root = Bone(id="b0", head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(root)

    # 1. Base reference wave (starts at t=0, wave cycle phi_t=0)
    clip_base = AnimationClip(name="Base", duration=2.0, armature=armature)
    clip_base.add_animation_movements(
        chain_wave_generator,
        list_bones=[0],
        wave_amplitude=0.3,
        wave_duration=2.0,
        frame_rate=10.0,
        phi_t=0.0,
        wave="travelling",
    )
    anim_base = clip_base.generate_animation()

    # 2. phi_t = -pi/2 (Quarter-cycle temporal phase shift: sin(ks - omega*t - pi/2) at t=0 matches t=0.5s)
    clip_phase = AnimationClip(name="PhaseOffset", duration=2.0, armature=armature)
    clip_phase.add_animation_movements(
        chain_wave_generator,
        list_bones=[0],
        wave_amplitude=0.3,
        wave_duration=2.0,
        frame_rate=10.0,
        phi_t=-np.pi / 2,
        wave="travelling",
    )
    anim_phase = clip_phase.generate_animation()

    # Timeline bounds must be preserved at [0.0, 2.0)
    assert min(anim_phase.keys()) == 0.0
    # At timeline t=0, frame matches base at t=0.5s
    np.testing.assert_allclose(
        anim_phase[0.0][0].numpy(),
        anim_base[0.5][0].numpy(),
        atol=1e-4,
    )

    # 3. timeline_offset = 0.5s (Delays the movement to start at t=0.5 on the timeline)
    clip_tl_offset = AnimationClip(
        name="TimelineOffset", duration=2.0, armature=armature
    )
    clip_tl_offset.add_animation_movements(
        chain_wave_generator,
        list_bones=[0],
        wave_amplitude=0.3,
        wave_duration=2.0,
        frame_rate=10.0,
        timeline_offset=0.5,
        wave="travelling",
    )
    anim_tl_offset = clip_tl_offset.generate_animation()

    # Timeline keys start at 0.5s
    assert min(anim_tl_offset.keys()) == 0.5
    # At timeline t=0.5, frame matches base at t=0.0
    np.testing.assert_allclose(
        anim_tl_offset[0.5][0].numpy(),
        anim_base[0.0][0].numpy(),
        atol=1e-4,
    )
