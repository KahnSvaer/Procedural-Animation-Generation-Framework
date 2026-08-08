import pytest
import trimesh
import numpy as np
import torch
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from pathlib import Path

from animgen.rigging.mesh_contraction import extract_skeleton
from animgen.core.spline import Spline
from animgen.renderer.visualizations import (
    visualize_skeleton_over_mesh,
    visualize_skeleton,
)


def test_skeleton_extraction_on_cylinder():
    mesh = trimesh.creation.cylinder(radius=1.0, height=10.0, sections=16)

    mesh = mesh.subdivide()
    mesh = mesh.subdivide()

    skeleton = extract_skeleton(mesh, max_iters=5, epsilon=1e-5)

    assert isinstance(skeleton, trimesh.path.Path3D)
    assert len(skeleton.vertices) > 0
    assert len(skeleton.entities) > 0

    mean_coords = np.mean(skeleton.vertices, axis=0)
    assert np.abs(mean_coords[0]) < 0.5
    assert np.abs(mean_coords[1]) < 0.5

    z_min, z_max = np.min(skeleton.vertices[:, 2]), np.max(skeleton.vertices[:, 2])
    assert z_min < -3.0
    assert z_max > 3.0


TEST_CURVE_DATASETS = [
    # Dataset 1: S-Curve
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 1.0],
        [1.0, 0.8, 2.0],
        [0.8, 1.5, 3.0],
        [0.0, 1.8, 4.0],
        [-0.8, 1.2, 5.0],
        [-1.0, 0.0, 6.0],
    ],
    # Dataset 2: Helix / Spiral
    [
        [1.0, 0.0, 0.0],
        [0.7, 0.7, 1.0],
        [0.0, 1.0, 2.0],
        [-0.7, 0.7, 3.0],
        [-1.0, 0.0, 4.0],
        [-0.7, -0.7, 5.0],
        [0.0, -1.0, 6.0],
    ],
    # Dataset 3: C-Curve Arch
    [
        [0.0, 0.0, 0.0],
        [0.8, 0.0, 0.5],
        [1.5, 0.0, 1.5],
        [2.0, 0.0, 3.0],
        [1.5, 0.0, 4.5],
        [0.8, 0.0, 5.5],
        [0.0, 0.0, 6.0],
    ],
    # Dataset 4: Zig-Zag Wave
    [
        [0.0, 0.0, 0.0],
        [0.6, 0.6, 1.0],
        [0.0, 1.2, 2.0],
        [-0.6, 0.6, 3.0],
        [0.0, 0.0, 4.0],
        [0.6, -0.6, 5.0],
        [0.0, -1.2, 6.0],
    ],
]


@pytest.mark.parametrize("control_points", TEST_CURVE_DATASETS)
def test_skeleton_extraction_on_catmull_rom_tube(control_points):
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in control_points]

    spline = Spline(pts, alpha=0.5)
    curve_tensors = spline.evaluate_curve(num_points_per_segment=10)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius = 0.3
    circle_poly = Point(0, 0).buffer(radius, quad_segs=8)
    mesh = trimesh.creation.sweep_polygon(circle_poly, curve_pts)

    skeleton = extract_skeleton(mesh, max_iters=10, epsilon=1e-5, no_1d_collapses=True)

    ds_idx = TEST_CURVE_DATASETS.index(control_points) + 1
    save_dirs = Path("tests/artifacts/mesh_contraction")
    save_dirs.mkdir(parents=True, exist_ok=True)
    mesh.export(str(save_dirs / f"dataset_{ds_idx}_tube_mesh.glb"))
    print(ds_idx, skeleton.vertices)

    visualize_skeleton(skeleton).export(
        str(save_dirs / f"dataset_{ds_idx}_skeleton.glb")
    )
    visualize_skeleton_over_mesh(mesh, skeleton).export(
        str(save_dirs / f"dataset_{ds_idx}_skeleton_over_mesh.glb")
    )

    assert isinstance(skeleton, trimesh.path.Path3D), "Skeleton is not a Path3D object."
    assert len(skeleton.vertices) > 0, (
        f"Skeleton has {len(skeleton.vertices)} vertices."
    )
    assert len(skeleton.entities) > 0, (
        f"Skeleton has {len(skeleton.entities)} entities."
    )

    dists = cdist(skeleton.vertices, curve_pts)
    min_dists = np.min(dists, axis=1)
    mean_error = np.mean(min_dists)

    assert mean_error < radius * 0.6, (
        f"Skeleton error {mean_error:.4f} exceeds threshold!"
    )
