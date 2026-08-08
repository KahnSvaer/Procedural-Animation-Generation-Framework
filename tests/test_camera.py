import torch

from animgen.utils.camera import (
    matrix3x4_to_4x4,
    view_matrix,
    sample_view_matrices,
    sample_view_matrices_polyhedra,
)


def test_matrix3x4_to_4x4():
    """Test converting 3x4 transformation matrices to homogeneous 4x4."""
    mat_3x4 = torch.randn(2, 3, 4)
    mat_4x4 = matrix3x4_to_4x4(mat_3x4)

    assert mat_4x4.shape == (2, 4, 4)
    assert torch.allclose(mat_4x4[:, 3, :3], torch.zeros(2, 3))
    assert torch.allclose(mat_4x4[:, 3, 3], torch.ones(2))


def test_view_matrix_lookat():
    """Test generating camera view matrix looking at a center point."""
    cam_pos = torch.tensor([[0.0, 0.0, 5.0]])
    lookat = torch.tensor([[0.0, 0.0, 0.0]])
    up = torch.tensor([0.0, 1.0, 0.0])

    poses = view_matrix(cam_pos, lookat, up)
    assert poses.shape == (1, 4, 4)

    assert torch.allclose(poses[0, :3, 3], cam_pos[0])


def test_sample_view_matrices():
    """Test sampling view matrices on a sphere surface."""
    num_samples = 8
    radius = 4.0
    poses = sample_view_matrices(n=num_samples, radius=radius)

    assert poses.shape == (num_samples, 4, 4)
    for i in range(num_samples):
        pos = poses[i, :3, 3]
        dist = torch.linalg.vector_norm(pos)
        assert torch.allclose(dist, torch.tensor(radius))


def test_sample_view_matrices_polyhedra():
    """Test camera sampling positions based on regular polyhedra."""
    radius = 5.0

    # 1. Dodecahedron sampling (20 vertices)
    poses_dodeca = sample_view_matrices_polyhedra(polygon="dodecahedron", radius=radius)
    assert poses_dodeca.shape == (20, 4, 4)
    dists_dodeca = torch.linalg.vector_norm(poses_dodeca[:, :3, 3], dim=-1)
    assert torch.allclose(dists_dodeca, torch.tensor(radius))

    # 2. Icosahedron sampling (12 vertices)
    poses_icosa = sample_view_matrices_polyhedra(polygon="icosahedron", radius=radius)
    assert poses_icosa.shape == (12, 4, 4)
    dists_icosa = torch.linalg.vector_norm(poses_icosa[:, :3, 3], dim=-1)
    assert torch.allclose(dists_icosa, torch.tensor(radius))
