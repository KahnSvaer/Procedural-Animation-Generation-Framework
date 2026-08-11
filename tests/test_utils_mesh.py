import trimesh
import numpy as np

from animgen.utils.mesh import taubin_smoothing


def test_taubin_smoothing():
    mesh = trimesh.creation.cylinder(radius=1.0, height=5.0, sections=16)

    np.random.seed(42)
    noise = np.random.normal(scale=0.05, size=mesh.vertices.shape)
    mesh.vertices += noise

    dists_initial = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    dists_initial_body = dists_initial[dists_initial > 0.5]
    roughness_initial = np.std(dists_initial_body)

    smoothed = taubin_smoothing(mesh, lamb=0.5, nu=0.53, iterations=15)

    # Vertices and faces should remain the same size
    assert len(smoothed.vertices) == len(mesh.vertices)
    assert len(smoothed.faces) == len(mesh.faces)

    # Roughness should be significantly reduced
    dists_smoothed = np.linalg.norm(smoothed.vertices[:, :2], axis=1)
    dists_smoothed_body = dists_smoothed[dists_smoothed > 0.5]
    roughness_smoothed = np.std(dists_smoothed_body)

    print(f"Initial Roughness: {roughness_initial:.4f}")
    print(f"Smoothed Roughness: {roughness_smoothed:.4f}")

    assert roughness_smoothed < roughness_initial * 0.7

    # Volume/radius should NOT shrink significantly (Taubin non-shrinking property)
    mean_radius_smoothed = np.mean(dists_smoothed_body)
    print(f"Mean Smoothed Radius: {mean_radius_smoothed:.4f}")
    assert mean_radius_smoothed > 0.80
