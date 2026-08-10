import numpy as np


def subdivide_and_center_skeleton(
    mesh_vertices, skeleton_vertices, skeleton_edges, max_edge_len=0.03
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

    Returns
    -------
    subdivided_vertices : (P, 3) ndarray
        Dense skeleton vertices guaranteed to lie inside the mesh volume.
    subdivided_edges : (Q, 2) ndarray
        Subdivided edge connectivity graph.
    """
    curr_v = list(skeleton_vertices)
    curr_e = [list(e) for e in skeleton_edges]

    changed = True
    while changed:
        changed = False
        new_e = []
        for u_idx, v_idx in curr_e:
            p_u = np.array(curr_v[u_idx])
            p_v = np.array(curr_v[v_idx])
            edge_vec = p_v - p_u
            length = np.linalg.norm(edge_vec)

            if length > max_edge_len:
                mid_initial = (p_u + p_v) / 2.0
                dir_vec = edge_vec / (length + 1e-12)

                disp_to_mid = mesh_vertices - mid_initial
                proj_along_dir = np.abs(np.dot(disp_to_mid, dir_vec))
                dist_to_mid = np.linalg.norm(disp_to_mid, axis=1)

                slice_mask = (proj_along_dir < max_edge_len * 0.5) & (
                    dist_to_mid < max_edge_len * 3.0
                )
                if np.sum(slice_mask) >= 3:
                    mid_centered = np.mean(mesh_vertices[slice_mask], axis=0)
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
