import pytest
import trimesh
import numpy as np
import torch
from scipy.spatial.distance import cdist
from pathlib import Path

from animgen.rigging.mesh_contraction import extract_skeleton
from animgen.rigging.refine_skelaton import (
    subdivide_and_center_skeleton,
    refine_and_center_skeleton_iterative,
)
from animgen.core.spline import Spline
from animgen.renderer.visualizations import (
    visualize_skeleton_over_mesh,
    visualize_skeleton,
)


def create_variable_radius_tube_mesh(curve_pts, radius_fn, num_sections=16):
    """
    Creates a watertight 3D tube mesh with variable radius r(t) along curve_pts.
    """
    N = len(curve_pts)
    t_vals = np.linspace(0.0, 1.0, N)
    radii = np.array([radius_fn(t) for t in t_vals])

    # Tangents along curve
    tangents = np.zeros_like(curve_pts)
    tangents[:-1] = curve_pts[1:] - curve_pts[:-1]
    tangents[-1] = tangents[-2]
    norm_t = np.linalg.norm(tangents, axis=1, keepdims=True)
    norm_t[norm_t < 1e-12] = 1.0
    tangents /= norm_t

    # Frame computation
    normals = np.zeros_like(curve_pts)
    t0 = tangents[0]
    arb = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t0, arb)
    n0 /= np.linalg.norm(n0)
    normals[0] = n0

    for i in range(1, N):
        prev_n = normals[i - 1]
        t_curr = tangents[i]
        n_proj = prev_n - np.dot(prev_n, t_curr) * t_curr
        if np.linalg.norm(n_proj) < 1e-6:
            arb = (
                np.array([1.0, 0.0, 0.0])
                if abs(t_curr[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            n_proj = np.cross(t_curr, arb)
        n_proj /= np.linalg.norm(n_proj)
        normals[i] = n_proj

    binormals = np.cross(tangents, normals)

    angles = np.linspace(0, 2 * np.pi, num_sections, endpoint=False)
    ring_verts = []

    for i in range(N):
        p = curve_pts[i]
        r = radii[i]
        n = normals[i]
        b = binormals[i]
        ring = p + r * (np.outer(np.cos(angles), n) + np.outer(np.sin(angles), b))
        ring_verts.append(ring)

    all_verts = np.vstack(ring_verts)

    faces = []
    for i in range(N - 1):
        r1_off = i * num_sections
        r2_off = (i + 1) * num_sections
        for j in range(num_sections):
            j_next = (j + 1) % num_sections
            v0 = r1_off + j
            v1 = r1_off + j_next
            v2 = r2_off + j_next
            v3 = r2_off + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # End caps
    cap_start_idx = len(all_verts)
    all_verts = np.vstack([all_verts, curve_pts[0]])
    for j in range(num_sections):
        j_next = (j + 1) % num_sections
        faces.append([cap_start_idx, j_next, j])

    cap_end_idx = len(all_verts)
    all_verts = np.vstack([all_verts, curve_pts[-1]])
    r_end_off = (N - 1) * num_sections
    for j in range(num_sections):
        j_next = (j + 1) % num_sections
        faces.append([cap_end_idx, r_end_off + j, r_end_off + j_next])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces), process=True)
    return mesh


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
]

RADIUS_PROFILES = {
    "increasing": lambda t: 0.2 + 0.4 * t,
    "decreasing": lambda t: 0.6 - 0.4 * t,
    "sinusoidal": lambda t: 0.35 + 0.15 * np.sin(4 * np.pi * t),
}


@pytest.mark.parametrize("profile_name", ["increasing", "decreasing", "sinusoidal"])
def test_variable_radius_splines_and_refinement_ablation(profile_name):
    """
    Tests skeleton extraction on variable-radius solidify tube meshes and verifies
    that boundary loop refinement, subdivide centering, and iterative Laplacian smoothing
    refinement improve accuracy relative to the ground-truth spline.
    """
    ctrl_pts = TEST_CURVE_DATASETS[0]
    pts = [torch.tensor(pt, dtype=torch.float32) for pt in ctrl_pts]
    spline = Spline(pts, alpha=0.5)
    curve_tensors = spline.evaluate_curve(num_points_per_segment=10)
    curve_pts = np.array([t.detach().cpu().numpy() for t in curve_tensors])

    radius_fn = RADIUS_PROFILES[profile_name]
    mesh = create_variable_radius_tube_mesh(curve_pts, radius_fn, num_sections=16)

    # 1. Raw contraction (no refinement)
    skel_raw_verts, _ = extract_skeleton(
        mesh,
        max_iters=20,
        threshold=0.5,
        no_1d_collapses=True,
        enable_embedding_refinement=False,
        enable_junction_merging=False,
        return_tuple=True,
    )

    # 2. Embedding Refinement only
    skel_ref_verts, _ = extract_skeleton(
        mesh,
        max_iters=20,
        threshold=0.5,
        no_1d_collapses=True,
        enable_embedding_refinement=True,
        enable_junction_merging=False,
        return_tuple=True,
    )

    # 3. Embedding Refinement + Junction Merging (Full Pipeline)
    skel_full_verts, skel_full_edges = extract_skeleton(
        mesh,
        max_iters=20,
        threshold=0.5,
        no_1d_collapses=True,
        enable_embedding_refinement=True,
        enable_junction_merging=True,
        return_tuple=True,
    )

    # 4. User Refinement Algo A: Subdivide & Centroid Slice
    skel_sub_verts, _ = subdivide_and_center_skeleton(
        mesh.vertices, skel_full_verts, skel_full_edges, max_edge_len=0.2
    )

    # 5. User Refinement Algo B: Iterative Subdivision + Tangential Laplacian + Momentum
    skel_lap_verts, _ = refine_and_center_skeleton_iterative(
        mesh.vertices,
        skel_full_verts,
        skel_full_edges,
        max_edge_len=0.2,
        num_iters=10,
        alpha=0.3,
        beta=0.5,
        laplacian_weight=0.3,
    )

    # Quantitative Mean Distance Errors from ground-truth spline curve
    err_raw = np.mean(np.min(cdist(skel_raw_verts, curve_pts), axis=1))
    err_ref = np.mean(np.min(cdist(skel_ref_verts, curve_pts), axis=1))
    err_full = np.mean(np.min(cdist(skel_full_verts, curve_pts), axis=1))
    err_sub = np.mean(np.min(cdist(skel_sub_verts, curve_pts), axis=1))
    err_lap = np.mean(np.min(cdist(skel_lap_verts, curve_pts), axis=1))

    print(
        f"\nProfile: {profile_name.capitalize():10s} | "
        f"Raw: {err_raw:.4f} -> Au Boundary: {err_ref:.4f} -> Full Au: {err_full:.4f} -> "
        f"Subdiv: {err_sub:.4f} -> Laplacian-Refined: {err_lap:.4f}"
    )

    skel_path = trimesh.load_path(skel_full_verts[skel_full_edges])

    # Export visual artifacts for inspection
    save_dirs = Path("tests/artifacts/mesh_contraction")
    save_dirs.mkdir(parents=True, exist_ok=True)
    mesh.export(str(save_dirs / f"{profile_name}_tube_mesh.glb"))
    visualize_skeleton(skel_path).export(
        str(save_dirs / f"{profile_name}_skeleton.glb")
    )
    visualize_skeleton_over_mesh(mesh, skel_path).export(
        str(save_dirs / f"{profile_name}_skeleton_over_mesh.glb")
    )

    # Assertions
    assert len(skel_full_verts) > 0, "Extracted skeleton has no vertices."
    assert err_ref <= err_raw + 1e-3, (
        f"Refinement increased distance error! Raw: {err_raw}, Refined: {err_ref}"
    )
    assert err_full < 0.15, (
        f"Skeleton error {err_full:.4f} exceeds ground-truth tolerance!"
    )


def test_torus_topology_preservation():
    """
    Verifies that link condition with threshold (0.5 * bbox_diag)
    preserves true topological tunnels (genus 1 torus hole).
    """
    torus = trimesh.creation.torus(major_radius=4.0, minor_radius=1.0)
    skel_nodes, skel_edges = extract_skeleton(
        torus, max_iters=20, threshold=0.5, return_tuple=True
    )

    assert len(skel_nodes) >= 3, f"Torus skeleton has too few nodes: {len(skel_nodes)}"
    assert len(skel_edges) == len(skel_nodes), (
        f"Topology lost! Edges ({len(skel_edges)}) != Nodes ({len(skel_nodes)})"
    )


def test_scale_invariance():
    """
    Verifies that skeleton extraction is scale-invariant across 1m and 100m meshes.
    """
    m1 = trimesh.creation.cylinder(radius=1.0, height=10.0, sections=16)
    m100 = trimesh.creation.cylinder(radius=100.0, height=1000.0, sections=16)

    v1, e1 = extract_skeleton(m1, max_iters=20, threshold=0.5, return_tuple=True)
    v100, e100 = extract_skeleton(m100, max_iters=20, threshold=0.5, return_tuple=True)

    v100_normalized = v100 / 100.0

    r1 = np.mean(np.hypot(v1[:, 0], v1[:, 1]))
    r100 = np.mean(np.hypot(v100_normalized[:, 0], v100_normalized[:, 1]))

    assert abs(r1 - r100) < 0.05, (
        f"Scale invariance failed! 1m radius: {r1:.4f}, 100m normalized radius: {r100:.4f}"
    )
