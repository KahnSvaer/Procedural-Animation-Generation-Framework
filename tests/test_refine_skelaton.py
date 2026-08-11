import numpy as np
import trimesh
import torch
from shapely.geometry import Point

from animgen.rigging.refine_skelaton import (
    subdivide_and_center_skeleton,
    refine_and_center_skeleton_iterative,
)
from animgen.core.spline import Spline


def test_subdivide_and_center_skeleton():
    control_points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 1.0],
        [1.0, 0.8, 2.0],
        [0.8, 1.5, 3.0],
        [0.0, 1.8, 4.0],
    ]
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in control_points]
    spline = Spline(pts, alpha=0.5)
    curve_tensors = spline.evaluate_curve(num_points_per_segment=10)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius = 0.3
    circle_poly = Point(0, 0).buffer(radius, quad_segs=8)
    mesh = trimesh.creation.sweep_polygon(circle_poly, curve_pts)

    np.random.seed(42)
    offset_noise = np.random.normal(scale=0.08, size=curve_pts.shape)
    offset_noise[0] = 0
    offset_noise[-1] = 0
    noisy_skeleton_vertices = curve_pts + offset_noise

    num_verts = len(noisy_skeleton_vertices)
    skeleton_edges = np.column_stack(
        (np.arange(num_verts - 1), np.arange(1, num_verts))
    )

    max_edge_len = 0.1
    subdivided_vertices, subdivided_edges = subdivide_and_center_skeleton(
        mesh.vertices,
        noisy_skeleton_vertices,
        skeleton_edges,
        max_edge_len=max_edge_len,
    )

    for u, v in subdivided_edges:
        edge_len = np.linalg.norm(subdivided_vertices[u] - subdivided_vertices[v])
        assert edge_len <= max_edge_len + 1e-5


def test_refine_and_center_skeleton_iterative():
    control_points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 1.0],
        [1.0, 0.8, 2.0],
        [0.8, 1.5, 3.0],
        [0.0, 1.8, 4.0],
    ]
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in control_points]
    spline = Spline(pts, alpha=0.5)
    curve_tensors = spline.evaluate_curve(num_points_per_segment=10)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius = 0.3
    circle_poly = Point(0, 0).buffer(radius, quad_segs=8)
    mesh = trimesh.creation.sweep_polygon(circle_poly, curve_pts)

    np.random.seed(42)
    offset_noise = np.random.normal(scale=0.08, size=curve_pts.shape)
    offset_noise[0] = 0
    offset_noise[-1] = 0
    noisy_skeleton_vertices = curve_pts + offset_noise

    num_verts = len(noisy_skeleton_vertices)
    skeleton_edges = np.column_stack(
        (np.arange(num_verts - 1), np.arange(1, num_verts))
    )

    max_edge_len = 0.1
    refined_vertices, refined_edges = refine_and_center_skeleton_iterative(
        mesh.vertices,
        noisy_skeleton_vertices,
        skeleton_edges,
        max_edge_len=max_edge_len,
        num_iters=10,
        alpha=0.3,
        beta=0.5,
        laplacian_weight=0.9,
    )

    # Verify edge subdivision bounds
    for u, v in refined_edges:
        edge_len = np.linalg.norm(refined_vertices[u] - refined_vertices[v])
        assert edge_len <= max_edge_len + 1e-5

    # Verify that the average noise distance of the centered vertices to the original spline is smaller
    # than the initial noise distance (centering should pull them back to the center of the tube)
    initial_dists = np.linalg.norm(noisy_skeleton_vertices - curve_pts, axis=1)

    refined_dists_to_spline = []
    for pt in refined_vertices:
        dists = np.linalg.norm(curve_pts - pt, axis=1)
        refined_dists_to_spline.append(np.min(dists))

    mean_initial_dist = np.mean(initial_dists)
    mean_refined_dist = np.mean(refined_dists_to_spline)

    print(f"Mean initial distance to spline: {mean_initial_dist:.4f}")
    print(f"Mean refined distance to spline: {mean_refined_dist:.4f}")

    assert mean_refined_dist < mean_initial_dist * 0.75
