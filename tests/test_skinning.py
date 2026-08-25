import numpy as np
import pytest
import trimesh
from animgen.core.armature import Armature, Bone
from animgen.rigging.skinning import (
    compute_auto_skin_weights,
    get_skinning_weight_matrix,
    compute_robust_cotangent_laplacian,
    dist_point_to_segment_vectorized,
)


def test_dist_point_to_segment_vectorized():
    # Test point projection onto segment [0,0,0] to [0,0,1]
    pts = np.array(
        [
            [0.0, 0.0, 0.5],  # On segment
            [1.0, 0.0, 0.5],  # Offset by 1 in X at midpoint
            [0.0, 0.0, -1.0],  # Before start (clamped to start)
            [0.0, 0.0, 2.0],  # Beyond end (clamped to end)
        ]
    )
    seg_a = np.array([0.0, 0.0, 0.0])
    seg_b = np.array([0.0, 0.0, 1.0])

    dists, proj = dist_point_to_segment_vectorized(pts, seg_a, seg_b)
    np.testing.assert_allclose(dists, [0.0, 1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        proj,
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-6,
    )


def test_cotangent_laplacian_properties():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    L, M = compute_robust_cotangent_laplacian(mesh)
    N = len(mesh.vertices)

    assert L.shape == (N, N)
    assert M.shape == (N, N)

    # Row sum of Laplacian matrix should be approximately 0
    row_sums = np.array(L.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-5)

    # Mass matrix should be diagonal and strictly positive
    diag_m = M.diagonal()
    assert np.all(diag_m > 0)


def test_skinning_cylinder_connected_chain():
    # 3-bone chain along Z axis in a cylinder
    mesh = trimesh.creation.cylinder(radius=0.5, height=3.0, sections=24)
    b0 = Bone(id="bone_bottom", head=(0, 0, -1.5), tail=(0, 0, -0.5))
    armature = Armature(b0)
    b1 = armature.add_connected_bone(b0, tail=(0, 0, 0.5))
    b1.id = "bone_mid"
    b2 = armature.add_connected_bone(b1, tail=(0, 0, 1.5))
    b2.id = "bone_top"

    weights = compute_auto_skin_weights(mesh, armature)
    assert set(weights.keys()) == {"bone_bottom", "bone_mid", "bone_top"}

    matrix, bone_ids = get_skinning_weight_matrix(mesh, armature)
    assert matrix.shape == (len(mesh.vertices), 3)
    assert bone_ids == ["bone_bottom", "bone_mid", "bone_top"]

    # Invariants: weights must be in [0, 1] and rows must sum to 1.0
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
    row_sums = np.sum(matrix, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # Monotonicity checks along Z:
    # Vertices at z < -1.0 should have highest weight for bone_bottom
    bottom_verts = mesh.vertices[:, 2] < -1.0
    assert np.all(matrix[bottom_verts, 0] > matrix[bottom_verts, 1])
    assert np.all(matrix[bottom_verts, 0] > matrix[bottom_verts, 2])

    # Vertices at z > 1.0 should have highest weight for bone_top
    top_verts = mesh.vertices[:, 2] > 1.0
    assert np.all(matrix[top_verts, 2] > matrix[top_verts, 1])
    assert np.all(matrix[top_verts, 2] > matrix[top_verts, 0])


def test_skinning_disconnected_mesh_and_bones():
    # 2 disjoint boxes along X axis
    box_a = trimesh.creation.box(extents=(1, 1, 1))
    box_a.apply_translation([-3, 0, 0])
    box_b = trimesh.creation.box(extents=(1, 1, 1))
    box_b.apply_translation([3, 0, 0])
    combined_mesh = box_a + box_b

    # 2 disconnected bones: bone_a inside box_a, bone_b inside box_b
    root = Bone(id="bone_a", head=(-3, 0, -0.5), tail=(-3, 0, 0.5))
    armature = Armature(root)
    bone_b = armature.add_unconnected_bone(root, head=(3, 0, -0.5), tail=(3, 0, 0.5))
    bone_b.id = "bone_b"

    weights = compute_auto_skin_weights(combined_mesh, armature)

    # Box A vertices (X < 0) should be 100% attached to bone_a
    mask_a = combined_mesh.vertices[:, 0] < 0
    np.testing.assert_allclose(weights["bone_a"][mask_a], 1.0, atol=1e-4)
    np.testing.assert_allclose(weights["bone_b"][mask_a], 0.0, atol=1e-4)

    # Box B vertices (X > 0) should be 100% attached to bone_b
    mask_b = combined_mesh.vertices[:, 0] > 0
    np.testing.assert_allclose(weights["bone_b"][mask_b], 1.0, atol=1e-4)
    np.testing.assert_allclose(weights["bone_a"][mask_b], 0.0, atol=1e-4)


def test_skinning_branching_and_disconnected_armature():
    # Sphere with T-shape armature
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    root = Bone(id="pelvis", head=(0, 0, -0.8), tail=(0, 0, 0.0))
    armature = Armature(root)
    spine = armature.add_connected_bone(root, tail=(0, 0, 0.8))
    spine.id = "spine"
    arm_l = armature.add_unconnected_bone(spine, head=(0, 0, 0.4), tail=(-0.8, 0, 0.4))
    arm_l.id = "arm_l"
    arm_r = armature.add_unconnected_bone(spine, head=(0, 0, 0.4), tail=(0.8, 0, 0.4))
    arm_r.id = "arm_r"

    matrix, bone_ids = get_skinning_weight_matrix(sphere, armature)
    assert matrix.shape == (len(sphere.vertices), 4)

    # Check valid distribution
    assert np.all(matrix >= 0.0)
    row_sums = np.sum(matrix, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # Left lateral vertices (near arm_l at z ~ 0.4, x < -0.6)
    left_verts = (sphere.vertices[:, 0] < -0.6) & (sphere.vertices[:, 2] > 0.0)
    idx_l = bone_ids.index("arm_l")
    assert np.all(np.argmax(matrix[left_verts], axis=1) == idx_l)

    # Right lateral vertices (near arm_r at z ~ 0.4, x > 0.6)
    right_verts = (sphere.vertices[:, 0] > 0.6) & (sphere.vertices[:, 2] > 0.0)
    idx_r = bone_ids.index("arm_r")
    assert np.all(np.argmax(matrix[right_verts], axis=1) == idx_r)


def test_skinning_extreme_armatures_and_meshes():
    # Extreme armature configurations:
    # 1. Bones extending way outside mesh
    # 2. Zero-length bone (head == tail)
    box = trimesh.creation.box(extents=(1, 1, 1))

    b0 = Bone(id="huge_bone", head=(0, 0, -10.0), tail=(0, 0, 10.0))
    armature = Armature(b0)
    b_zero = armature.add_unconnected_bone(
        b0, head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.0)
    )
    b_zero.id = "zero_bone"

    weights = compute_auto_skin_weights(box, armature)
    assert len(weights["huge_bone"]) == len(box.vertices)
    assert len(weights["zero_bone"]) == len(box.vertices)

    W = np.column_stack([weights["huge_bone"], weights["zero_bone"]])
    assert np.all(W >= 0.0)
    np.testing.assert_allclose(np.sum(W, axis=1), 1.0, atol=1e-5)


def test_skinning_scene_input():
    # Pass a trimesh.Scene containing multiple geometries
    box = trimesh.creation.box()
    scene = trimesh.Scene(box)
    root = Bone(id="root", head=(0, 0, -0.5), tail=(0, 0, 0.5))
    armature = Armature(root)

    weights = compute_auto_skin_weights(scene, armature)
    assert "root" in weights
    assert len(weights["root"]) == len(scene.to_mesh().vertices)


def test_skinning_uv_seam_duplicate_vertices():
    # Cylinder with duplicated vertices along a seam
    cyl = trimesh.creation.cylinder(radius=0.5, height=3.0, sections=24)
    seam_mask = np.abs(cyl.vertices[:, 1] - 0.5) < 0.05
    seam_vert_indices = np.where(seam_mask)[0]

    dup_vertices = cyl.vertices[seam_vert_indices].copy()
    new_vertices = np.vstack([cyl.vertices, dup_vertices])
    new_faces = cyl.faces.copy()
    for idx_in_dup, orig_idx in enumerate(seam_vert_indices):
        new_idx = len(cyl.vertices) + idx_in_dup
        face_matches = np.where(new_faces == orig_idx)
        half = len(face_matches[0]) // 2
        for f_row, f_col in zip(face_matches[0][:half], face_matches[1][:half]):
            new_faces[f_row, f_col] = new_idx

    mesh_with_seam = trimesh.Trimesh(
        vertices=new_vertices, faces=new_faces, process=False
    )

    b0 = Bone(id="b0", head=(0, 0, -1.5), tail=(0, 0, -0.5))
    arm = Armature(b0)
    b1 = arm.add_connected_bone(b0, tail=(0, 0, 0.5))
    b1.id = "b1"
    b2 = arm.add_connected_bone(b1, tail=(0, 0, 1.5))
    b2.id = "b2"

    clean_w = compute_auto_skin_weights(cyl, arm)
    seam_w = compute_auto_skin_weights(mesh_with_seam, arm)

    for b_id in ["b0", "b1", "b2"]:
        # Duplicate vertices must have exact identical weights to avoid seam tearing
        np.testing.assert_allclose(
            seam_w[b_id][seam_vert_indices],
            seam_w[b_id][len(cyl.vertices) :],
            atol=1e-6,
        )
        # Original vertices should match untextured clean mesh
        np.testing.assert_allclose(
            seam_w[b_id][: len(cyl.vertices)],
            clean_w[b_id],
            atol=1e-4,
        )


def test_skinning_blender_correlation():
    # Compare pure NumPy Bone Heat solver with Blender ARMATURE_AUTO
    pytest.importorskip("bpy")

    mesh = trimesh.creation.cylinder(radius=0.5, height=3.0, sections=24)
    b0 = Bone(id="b0", head=(0, 0, -1.5), tail=(0, 0, -0.5))
    arm = Armature(b0)
    b1 = arm.add_connected_bone(b0, tail=(0, 0, 0.5))
    b1.id = "b1"
    b2 = arm.add_connected_bone(b1, tail=(0, 0, 1.5))
    b2.id = "b2"

    np_weights = compute_auto_skin_weights(mesh, arm, use_blender=False)
    bpy_weights = compute_auto_skin_weights(mesh, arm, use_blender=True)

    for b_id in ["b0", "b1", "b2"]:
        w_np = np_weights[b_id]
        w_bp = bpy_weights[b_id]
        diff = np.abs(w_np - w_bp)
        # Verify close agreement with Blender
        assert np.mean(diff) < 0.05
        assert np.max(diff) < 0.10
