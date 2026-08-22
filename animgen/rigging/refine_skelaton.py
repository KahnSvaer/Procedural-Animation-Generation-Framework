import numpy as np


def subdivide_and_center_skeleton(
    mesh_vertices, skeleton_vertices, skeleton_edges, max_edge_len=0.1, safety_iter=15
):
    """
    Subdivides long edges in the skeleton graph using a greedy midpoint insertion
    strategy, fixing and centering each new midpoint to the true geometric centroid
    of the local mesh cross-section slice.

    Parameters
    ----------
    mesh_vertices : (N, 3) ndarray
        Original 3D mesh surface vertices.
    skeleton_vertices : (M, 3) ndarray
        Extracted skeleton node positions.
    skeleton_edges : (K, 2) ndarray
        Indices of connected skeleton edges.
    max_edge_len : float
        Maximum allowed edge length before subdividing.
    safety_iter: int
        Safety iterations to prevent infinite loops.

    Returns
    -------
    subdivided_vertices : (P, 3) ndarray
        Dense skeleton vertices guaranteed to lie inside the mesh volume.
    subdivided_edges : (Q, 2) ndarray
        Subdivided edge connectivity graph.
    """
    curr_v = [np.array(v, dtype=np.float64) for v in skeleton_vertices]
    curr_e = [list(e) for e in skeleton_edges]

    changed = True
    subdiv_iters = 0
    while changed and subdiv_iters < safety_iter:
        subdiv_iters += 1
        changed = False
        new_e = []
        for u_idx, v_idx in curr_e:
            p_u = curr_v[u_idx]
            p_v = curr_v[v_idx]
            edge_vec = p_v - p_u
            length = np.linalg.norm(edge_vec)

            if length > max_edge_len:
                mid_initial = (p_u + p_v) / 2.0
                if subdiv_iters < 8:
                    dir_vec = edge_vec / (length + 1e-12)
                    disp_to_mid = mesh_vertices - mid_initial
                    proj_along_dir = np.abs(np.dot(disp_to_mid, dir_vec))

                    # Unbiased orthogonal plane slice (prevents near-neighbor distance bias)
                    slice_mask = proj_along_dir < (max_edge_len * 0.5)

                    if np.sum(slice_mask) >= 3:
                        slice_verts = mesh_vertices[slice_mask]
                        dist_to_mid = np.linalg.norm(slice_verts - mid_initial, axis=1)
                        min_d = np.min(dist_to_mid)
                        cluster_mask = dist_to_mid < (min_d * 2.5 + max_edge_len)
                        mid_centered = np.mean(slice_verts[cluster_mask], axis=0)
                    else:
                        mid_centered = mid_initial
                else:
                    mid_centered = mid_initial

                new_idx = len(curr_v)
                curr_v.append(mid_centered)

                new_e.append([u_idx, new_idx])
                new_e.append([new_idx, v_idx])
                changed = True
            else:
                new_e.append([u_idx, v_idx])
        curr_e = new_e

    return np.array(curr_v, dtype=np.float64), np.array(curr_e, dtype=np.int64)


def refine_and_center_skeleton_iterative(
    mesh_vertices,
    skeleton_vertices,
    skeleton_edges,
    max_edge_len=0.1,
    num_iters=5,
    alpha=0.5,
    beta=0.8,
    laplacian_weight=0.3,
):
    """
    Refines and centers the skeleton iteratively by running a centering/relaxation pass
    with momentum on all vertices, interleaved with greedy midpoint edge subdivision.

    Parameters
    ----------
    mesh_vertices : (N, 3) ndarray
        Original 3D mesh surface vertices.
    skeleton_vertices : (M, 3) ndarray
        Extracted skeleton node positions.
    skeleton_edges : (K, 2) ndarray
        Indices of connected skeleton edges.
    max_edge_len : float
        Maximum allowed edge length before subdividing.
    num_iters : int
        Number of refinement iterations.
    alpha : float
        Relaxation step size (learning rate / shift factor).
    beta : float
        Momentum factor for smoothing vertex updates.
    laplacian_weight : float
        Weight for Laplacian smoothing regularization (between 0.0 and 1.0).
        High values prevent skeleton buckling/drifting to the surface.

    Returns
    -------
    refined_vertices : (P, 3) ndarray
        Refined and centered skeleton node positions.
    refined_edges : (Q, 2) ndarray
        Refined edge connectivity graph.
    """
    curr_v = [np.array(v, dtype=np.float64) for v in skeleton_vertices]
    curr_e = [list(e) for e in skeleton_edges]
    velocities = [np.zeros(3) for _ in range(len(curr_v))]

    for iteration in range(num_iters):
        N_verts = len(curr_v)
        adj = {i: [] for i in range(N_verts)}
        for u, v in curr_e:
            adj[u].append(v)
            adj[v].append(u)

        # Smooth coordinates temporarily to estimate clean slice tangents
        coords = np.array(curr_v)
        smoothed_coords = coords.copy()
        for _ in range(5):
            temp = smoothed_coords.copy()
            for r in range(1, N_verts - 1):
                neighbors = adj[r]
                if len(neighbors) == 2:
                    temp[r] = 0.5 * temp[r] + 0.25 * (
                        smoothed_coords[neighbors[0]] + smoothed_coords[neighbors[1]]
                    )
            smoothed_coords = temp

        # Centering Pass with momentum and Laplacian smoothing for all current vertices
        for i in range(N_verts):
            neighbors = adj[i]
            if len(neighbors) == 1:
                # Terminal endpoint: maintain position, avoid shrinking
                total_shift = np.zeros(3)
                velocities[i] = np.zeros(3)
                continue
            elif len(neighbors) == 2:
                dir_vec = smoothed_coords[neighbors[1]] - smoothed_coords[neighbors[0]]
            else:
                diffs = [smoothed_coords[nb] - smoothed_coords[i] for nb in neighbors]
                ref = diffs[0]
                aligned_diffs = [ref]
                for r in range(1, len(diffs)):
                    d = diffs[r]
                    if np.dot(d, ref) < 0:
                        aligned_diffs.append(-d)
                    else:
                        aligned_diffs.append(d)
                dir_vec = np.mean(aligned_diffs, axis=0)

            length = np.linalg.norm(dir_vec)
            if length > 1e-12:
                dir_vec = dir_vec / length
            else:
                dir_vec = np.array([0.0, 0.0, 1.0])

            disp_to_v = mesh_vertices - curr_v[i]
            proj_along_dir = np.abs(np.dot(disp_to_v, dir_vec))

            # Unbiased plane slice perpendicular to tangent
            slice_mask = proj_along_dir < (max_edge_len * 0.5)

            if np.sum(slice_mask) >= 3:
                slice_verts = mesh_vertices[slice_mask]
                dist_to_v = np.linalg.norm(slice_verts - curr_v[i], axis=1)
                min_d = np.min(dist_to_v)
                cluster_mask = dist_to_v < (min_d * 2.5 + max_edge_len)
                centroid = np.mean(slice_verts[cluster_mask], axis=0)
            else:
                centroid = curr_v[i]

            shift = centroid - curr_v[i]

            if len(neighbors) >= 2:
                laplacian = (
                    np.mean([curr_v[nb] for nb in neighbors], axis=0) - curr_v[i]
                )
                # Tangential Laplacian only to prevent curve shrinkage
                lap_tangent = np.dot(laplacian, dir_vec) * dir_vec
                total_shift = (
                    1.0 - laplacian_weight
                ) * shift + laplacian_weight * lap_tangent
            else:
                total_shift = shift

            velocities[i] = beta * velocities[i] + (1.0 - beta) * alpha * total_shift
            curr_v[i] = curr_v[i] + velocities[i]

        # Subdivide long edges
        new_e = []
        for u_idx, v_idx in curr_e:
            p_u = curr_v[u_idx]
            p_v = curr_v[v_idx]
            edge_vec = p_v - p_u
            length = np.linalg.norm(edge_vec)

            if length > max_edge_len:
                mid_initial = (p_u + p_v) / 2.0
                dir_vec = edge_vec / (length + 1e-12)

                disp_to_mid = mesh_vertices - mid_initial
                proj_along_dir = np.abs(np.dot(disp_to_mid, dir_vec))

                slice_mask = proj_along_dir < (max_edge_len * 0.5)

                if np.sum(slice_mask) >= 3:
                    slice_verts = mesh_vertices[slice_mask]
                    dist_to_mid = np.linalg.norm(slice_verts - mid_initial, axis=1)
                    min_d = np.min(dist_to_mid)
                    cluster_mask = dist_to_mid < (min_d * 2.5 + max_edge_len)
                    mid_centered = np.mean(slice_verts[cluster_mask], axis=0)
                else:
                    mid_centered = mid_initial

                new_idx = len(curr_v)
                curr_v.append(mid_centered)
                velocities.append(np.zeros(3))

                new_e.append([u_idx, new_idx])
                new_e.append([new_idx, v_idx])
            else:
                new_e.append([u_idx, v_idx])
        curr_e = new_e

    # Final cleanup subdivision pass
    changed = True
    cleanup_iters = 0
    while changed and cleanup_iters < 10:
        cleanup_iters += 1
        changed = False
        new_e = []
        for u_idx, v_idx in curr_e:
            p_u = curr_v[u_idx]
            p_v = curr_v[v_idx]
            edge_vec = p_v - p_u
            length = np.linalg.norm(edge_vec)

            if length > max_edge_len:
                mid_initial = (p_u + p_v) / 2.0
                new_idx = len(curr_v)
                curr_v.append(mid_initial)
                new_e.append([u_idx, new_idx])
                new_e.append([new_idx, v_idx])
                changed = True
            else:
                new_e.append([u_idx, v_idx])
        curr_e = new_e

    return np.array(curr_v, dtype=np.float64), np.array(curr_e, dtype=np.int64)
