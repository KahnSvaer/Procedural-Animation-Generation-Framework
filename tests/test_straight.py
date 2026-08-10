import numpy as np
import torch
import trimesh
from shapely.geometry import Point

from animgen.animation.straight import straighten, straighten_lateral
from animgen.core.spline import Spline


def test_straighten_with_explicit_spine_points():
    """Test straightening a curved tube using explicit numpy spine points."""
    control_points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 1.0],
        [1.0, 0.8, 2.0],
        [0.8, 1.5, 3.0],
        [0.0, 1.8, 4.0],
    ]
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in control_points]
    spline = Spline(pts, alpha=0.5)
    curve_tensors = spline.evaluate_curve(num_points_per_segment=15)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius = 0.4
    circle_poly = Point(0, 0).buffer(radius, quad_segs=8)
    mesh = trimesh.creation.sweep_polygon(circle_poly, curve_pts)

    straightened_z = straighten(mesh, spine_points=curve_pts, axis="z")

    dists_xy = np.linalg.norm(straightened_z.vertices[:, :2], axis=1)
    assert np.max(dists_xy) < radius * 1.2
    assert np.mean(dists_xy) > radius * 0.75

    diffs = np.diff(curve_pts, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    z_coords = straightened_z.vertices[:, 2]
    assert np.min(z_coords) >= -0.25
    assert np.max(z_coords) <= total_length + 0.25

    straightened_x = straighten(mesh, spine_points=curve_pts, axis="x")
    dists_yz = np.linalg.norm(straightened_x.vertices[:, 1:3], axis=1)
    assert np.max(dists_yz) < radius * 1.2
    x_coords = straightened_x.vertices[:, 0]
    assert np.min(x_coords) >= -0.25
    assert np.max(x_coords) <= total_length + 0.25


def test_straighten_with_spline_object():
    """Test straightening using a Spline object directly as the spine."""
    control_points = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [2.0, 0.0, 3.0],
        [2.0, 0.0, 4.0],
    ]
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in control_points]
    spline = Spline(pts, alpha=0.5)

    curve_tensors = spline.evaluate_curve(num_points_per_segment=10)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius = 0.3
    circle_poly = Point(0, 0).buffer(radius, quad_segs=6)
    mesh = trimesh.creation.sweep_polygon(circle_poly, curve_pts)

    straightened = straighten(mesh, spine_points=spline, axis="y")

    dists_xz = np.linalg.norm(straightened.vertices[:, [0, 2]], axis=1)
    assert np.max(dists_xz) < radius * 1.2
    y_coords = straightened.vertices[:, 1]
    diffs = np.diff(curve_pts, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    assert np.min(y_coords) >= -0.25
    assert np.max(y_coords) <= total_length + 0.25


def test_straighten_lateral():
    """Test that straighten_lateral flattens X but preserves Y-Z curvature."""
    control_points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 1.0],
        [1.0, 0.5, 2.0],
        [0.5, 1.0, 3.0],
        [0.0, 1.5, 4.0],
    ]
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in control_points]
    spline = Spline(pts, alpha=0.5)
    curve_tensors = spline.evaluate_curve(num_points_per_segment=10)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius = 0.2
    circle_poly = Point(0, 0).buffer(radius, quad_segs=6)
    mesh = trimesh.creation.sweep_polygon(circle_poly, curve_pts)

    straightened = straighten_lateral(mesh, spine_points=curve_pts, straighten_axis="x")

    orig_var = np.var(mesh.vertices, axis=0)
    straight_var = np.var(straightened.vertices, axis=0)

    assert straight_var[0] < orig_var[0] * 0.4
    assert straight_var[1] > orig_var[1] * 0.8
    assert straight_var[2] > orig_var[2] * 0.8
