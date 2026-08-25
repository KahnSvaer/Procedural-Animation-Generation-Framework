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

    assert roughness_smoothed < roughness_initial * 0.7

    # Volume/radius should NOT shrink significantly (Taubin non-shrinking property)
    mean_radius_smoothed = np.mean(dists_smoothed_body)
    assert mean_radius_smoothed > 0.80


def test_mesh_differential_geometry_utils():
    from animgen.utils.mesh import (
        triangle_areas,
        vertex_areas,
        cotangent,
        compute_cotangent_laplacian,
    )

    # Simple equilateral-like triangle in XY plane
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    faces = np.array([[0, 1, 2]])

    areas = triangle_areas(verts, faces)
    expected_area = 0.5 * 1.0 * (np.sqrt(3) / 2)
    np.testing.assert_allclose(areas, [expected_area], atol=1e-6)

    v_areas = vertex_areas(verts, faces)
    np.testing.assert_allclose(v_areas, expected_area / 3.0, atol=1e-6)

    # Cotangent of 90 degrees should be 0, 45 degrees should be 1
    u = np.array([1.0, 0.0, 0.0])
    v_90 = np.array([0.0, 1.0, 0.0])
    v_45 = np.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(cotangent(u, v_90), 0.0, atol=1e-6)
    np.testing.assert_allclose(cotangent(u, v_45), 1.0, atol=1e-6)

    # Laplacian test with Trimesh object
    sphere = trimesh.creation.icosphere(subdivisions=2)
    L_only = compute_cotangent_laplacian(sphere, return_mass_matrix=False)
    L, M = compute_cotangent_laplacian(sphere, return_mass_matrix=True)

    assert L.shape == (len(sphere.vertices), len(sphere.vertices))
    assert M.shape == (len(sphere.vertices), len(sphere.vertices))
    np.testing.assert_allclose(L.toarray(), L_only.toarray(), atol=1e-6)
    # Row sum of L should be 0
    np.testing.assert_allclose(np.array(L.sum(axis=1)).flatten(), 0.0, atol=1e-5)
