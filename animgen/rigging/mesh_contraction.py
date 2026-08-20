"""
Mesh contraction and skeleton extraction code adapted from Au et al. (2008),
"Skeleton Extraction by Mesh Contraction" (ACM TOG 27(3):44).

This implementation is a Python adaptation of the original C++ code, with
modifications for clarity and integration.

References
----------
.. [1] Au, Oscar Kin-Chung, Tai, Chiew-Lan, Chu, Hung-Kuo, Cohen-Or, Daniel,
   and Lee, Tong-Yee. "Skeleton extraction by mesh contraction." ACM TOG 27(3):44, 2008.
   https://doi.org/10.1145/1360612.1360643
"""

import trimesh
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
import heapq


def triangle_areas(vertices, faces):
    """
    Compute the area of every triangle.

    Parameters
    ----------
    vertices : (N, 3) ndarray
    faces : (M, 3) ndarray

    Returns
    -------
    areas : (M,) ndarray
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def vertex_areas(vertices, faces):
    """
    Compute one-third barycentric area for every vertex.

    Parameters
    ----------
    vertices : (N, 3) ndarray
    faces : (M, 3) ndarray

    Returns
    -------
    areas : (N,) ndarray
    """
    face_areas = triangle_areas(vertices, faces)
    v_areas = np.zeros(len(vertices), dtype=np.float64)

    np.add.at(v_areas, faces[:, 0], face_areas / 3.0)
    np.add.at(v_areas, faces[:, 1], face_areas / 3.0)
    np.add.at(v_areas, faces[:, 2], face_areas / 3.0)

    return v_areas


def cotangent(u, v):
    """
    Compute cot(theta) between vectors u and v.
    """
    cross = np.cross(u, v)
    cross_norm = np.linalg.norm(cross, axis=-1)
    dot = np.sum(u * v, axis=-1)
    cot_val = dot / np.maximum(cross_norm, 1e-12)
    return np.clip(cot_val, -1e4, 1e4)


def cotangent_laplacian(vertices, faces):
    """
    Compute cotangent-weighted Laplace-Beltrami operator.
    """
    n = len(vertices)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cot0 = cotangent(v1 - v0, v2 - v0)
    cot1 = cotangent(v2 - v1, v0 - v1)
    cot2 = cotangent(v0 - v2, v1 - v2)

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    rows = np.concatenate([i, j, j, k, k, i])
    cols = np.concatenate([j, i, k, j, i, k])

    data = 0.5 * np.concatenate([cot2, cot2, cot0, cot0, cot1, cot1])

    L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    diag = np.asarray(L.sum(axis=1)).ravel()
    L = L - diags(diag)

    return L


def contraction_step(vertices, faces, WL, WH, use_pytorch=False, device=None):
    """
    Perform one geometry contraction step using SciPy or PyTorch sparse solver.
    """
    L = cotangent_laplacian(vertices, faces)

    # Weight matrices
    WL2 = diags(WL**2)
    WH2 = diags(WH**2)

    # Normal equations: (L.T @ WL2 @ L + WH2) V' = WH2 @ V
    M = L.T @ WL2 @ L + WH2
    RHS = WH2 @ vertices

    if use_pytorch:
        import torch

        dev = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        M_coo = M.tocoo()
        indices = torch.tensor(
            np.vstack((M_coo.row, M_coo.col)), dtype=torch.long, device=dev
        )
        values = torch.tensor(M_coo.data, dtype=torch.float32, device=dev)
        M_t = torch.sparse_coo_tensor(
            indices, values, M_coo.shape, device=dev
        ).to_dense()
        RHS_t = torch.tensor(RHS, dtype=torch.float32, device=dev)
        V_t = torch.linalg.solve(M_t, RHS_t)
        return V_t.cpu().numpy()

    return spsolve(M, RHS)


def contract_mesh(
    vertices,
    faces,
    max_iters=20,
    epsilon=1e-6,
    wl0=None,
    max_wh=100.0,
    use_pytorch=False,
    device=None,
):
    """
    Geometry contraction using constrained Laplacian smoothing (Au et al. Section 4).
    Supports SciPy CPU sparse solver (default) and PyTorch GPU solver (use_pytorch=True).
    """
    V = vertices.copy()
    F = faces.copy()
    N = len(V)

    # Bounding box diagonal scale S
    bbox_diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    if bbox_diag < 1e-12:
        bbox_diag = 1.0

    face_areas = triangle_areas(V, F)
    A = face_areas.mean()

    # Scale-invariant initial WL weight ratioing (Section 4)
    if wl0 is None:
        A_rel = A / (bbox_diag**2 + 1e-12)
        wl_val = 1e-3 * np.sqrt(A_rel)
        if wl_val < 1e-6:
            wl_val = 1.0
        WL = np.full(N, wl_val)
    else:
        WL = np.full(N, float(wl0))

    WH = np.ones(N)
    A0 = np.maximum(vertex_areas(V, F), 1e-12)

    # Calculate volume of original mesh
    try:
        vol0 = trimesh.Trimesh(V, F, process=False).volume
    except Exception:
        vol0 = 1.0
    if abs(vol0) < 1e-12:
        vol0 = 1.0

    for i in range(max_iters):
        V = contraction_step(V, F, WL, WH, use_pytorch=use_pytorch, device=device)
        At = vertex_areas(V, F)

        WL *= 2.0
        # Clamp WH to prevent ill-conditioned system explosion (max_wh)
        WH = np.minimum(np.sqrt(A0 / np.maximum(At, 1e-12)), max_wh)

        try:
            vol = trimesh.Trimesh(V, F, process=False).volume
        except Exception:
            vol = 0.0
        if abs(vol / vol0) < epsilon:
            break

    return V


def check_link_condition(
    i,
    j,
    vertex_neighbors,
    faces,
    vertex_faces,
    contracted_vertices=None,
    threshold=0.02,
):
    """
    Checks the topological Link Condition to ensure edge collapse preserves topology.
    Relaxes the condition for small loops (perimeter <= threshold) to allow collapsing
    cross-sections of thin tubes/cylinders in the zero-volume contracted mesh.
    """
    common_neighbors = vertex_neighbors[i].intersection(vertex_neighbors[j])
    for k in common_neighbors:
        shared_faces = (
            vertex_faces[i].intersection(vertex_faces[j]).intersection(vertex_faces[k])
        )
        if not shared_faces:
            if contracted_vertices is not None:
                vi = contracted_vertices[i]
                vj = contracted_vertices[j]
                vk = contracted_vertices[k]
                perimeter = (
                    np.linalg.norm(vi - vj)
                    + np.linalg.norm(vj - vk)
                    + np.linalg.norm(vk - vi)
                )
                if perimeter > threshold:
                    return False
            else:
                return False
    return True


def compute_edge_quadric(v_i, v_k):
    """
    Compute K_ik.T @ K_ik for the line of edge (i, k) using homogeneous QEM representation.
    """
    diff = v_k - v_i
    dist = np.linalg.norm(diff)
    if dist < 1e-12:
        return np.zeros((4, 4))
    a = diff / dist
    b = np.cross(a, v_i)

    K = np.zeros((3, 4))
    K[0, 1] = -a[2]
    K[0, 2] = a[1]
    K[0, 3] = -b[0]

    K[1, 0] = a[2]
    K[1, 2] = -a[0]
    K[1, 3] = -b[1]

    K[2, 0] = -a[1]
    K[2, 1] = a[0]
    K[2, 3] = -b[2]

    return K.T @ K


def compute_collapse_cost(i, j, vertices, vertex_neighbors, Q):
    """
    Computes total collapse cost F(i, j) = wa * Fa(i, j) + wb * Fb(i, j).
    """
    # 1. Shape Cost (Fa)
    Q_sum = Q[i] + Q[j]
    p_j = np.append(vertices[j], 1.0)
    F_a = p_j.T @ Q_sum @ p_j

    # 2. Sampling Cost (Fb)
    dist_ij = np.linalg.norm(vertices[i] - vertices[j])
    sum_dist_ik = sum(
        np.linalg.norm(vertices[i] - vertices[k]) for k in vertex_neighbors[i]
    )
    F_b = dist_ij * sum_dist_ik

    return F_a + 0.1 * F_b


def find_node(x, parent):
    """
    DSU Find with path compression.
    """
    curr = x
    path = []
    while parent[curr] != curr:
        path.append(curr)
        curr = parent[curr]
    for node in path:
        parent[node] = curr
    return curr


def connectivity_surgery(
    original_faces, contracted_vertices, threshold=0.5, no_1d_collapses=False
):
    """
    Performs Connectivity Surgery (Section 5) by simplifying the degenerate mesh
    using edge collapses until 0 active faces remain.
    """
    num_vertices = len(contracted_vertices)
    bbox_diag = np.linalg.norm(
        contracted_vertices.max(axis=0) - contracted_vertices.min(axis=0)
    )
    if threshold < 2.0 and bbox_diag > 1e-6:
        # Scale relative threshold by bounding box diagonal (Au et al. Section 5)
        threshold = threshold * bbox_diag

    faces = [list(f) for f in original_faces]
    active_faces = set(range(len(faces)))

    vertex_faces = {v: set() for v in range(num_vertices)}
    for f_idx, f in enumerate(faces):
        for v in f:
            vertex_faces[v].add(f_idx)

    vertex_neighbors = {v: set() for v in range(num_vertices)}
    for f in faces:
        vertex_neighbors[f[0]].add(f[1])
        vertex_neighbors[f[0]].add(f[2])
        vertex_neighbors[f[1]].add(f[0])
        vertex_neighbors[f[1]].add(f[2])
        vertex_neighbors[f[2]].add(f[0])
        vertex_neighbors[f[2]].add(f[1])

    # Initialize error matrices Q_i
    Q = {}
    for v in range(num_vertices):
        Q[v] = np.zeros((4, 4))
        for neighbor in vertex_neighbors[v]:
            Q[v] += compute_edge_quadric(
                contracted_vertices[v], contracted_vertices[neighbor]
            )

    # DSU to track skeleton-mesh mapping
    parent = list(range(num_vertices))

    # Initialize heap with edge collapses
    heap = []
    for u in range(num_vertices):
        for v in vertex_neighbors[u]:
            cost = compute_collapse_cost(u, v, contracted_vertices, vertex_neighbors, Q)
            heapq.heappush(heap, (cost, u, v))

    while len(active_faces) > 0 and heap:
        cost, i, j = heapq.heappop(heap)

        if i not in vertex_neighbors or j not in vertex_neighbors:
            continue
        if j not in vertex_neighbors[i]:
            continue

        # Prohibit collapsing 1D edges (edges that do not share any active faces)
        if no_1d_collapses:
            shared_faces = vertex_faces[i].intersection(vertex_faces[j])
            if len(shared_faces) == 0:
                continue

        if not check_link_condition(
            i, j, vertex_neighbors, faces, vertex_faces, contracted_vertices, threshold
        ):
            continue

        current_cost = compute_collapse_cost(
            i, j, contracted_vertices, vertex_neighbors, Q
        )
        if abs(current_cost - cost) > 1e-6:
            heapq.heappush(heap, (current_cost, i, j))
            continue

        # Perform collapse i -> j
        parent[i] = j

        # Update faces
        faces_to_update = list(vertex_faces[i])
        for f_idx in faces_to_update:
            face = faces[f_idx]
            vertex_faces[i].remove(f_idx)

            if j in face:
                for vertex in face:
                    if vertex != i and f_idx in vertex_faces[vertex]:
                        vertex_faces[vertex].remove(f_idx)
                if f_idx in active_faces:
                    active_faces.remove(f_idx)
            else:
                face[face.index(i)] = j
                vertex_faces[j].add(f_idx)

        # Update neighbors
        neighbors_to_update = list(vertex_neighbors[i])
        for k in neighbors_to_update:
            vertex_neighbors[k].discard(i)
            if k != j:
                vertex_neighbors[k].add(j)
                vertex_neighbors[j].add(k)

        vertex_neighbors[j].discard(i)

        # Delete vertex i
        del vertex_neighbors[i]
        del vertex_faces[i]

        # Update Q_j
        Q[j] = Q[i] + Q[j]

        # Push updated costs for neighbors of j
        for k in vertex_neighbors[j]:
            cost_kj = compute_collapse_cost(
                k, j, contracted_vertices, vertex_neighbors, Q
            )
            heapq.heappush(heap, (cost_kj, k, j))
            cost_jk = compute_collapse_cost(
                j, k, contracted_vertices, vertex_neighbors, Q
            )
            heapq.heappush(heap, (cost_jk, j, k))

    # Remaining active vertices form skeleton nodes
    skeletal_nodes = sorted(list(vertex_neighbors.keys()))

    # Build unique skeleton edges from remaining neighbor graph
    skeletal_edges = []
    for u in skeletal_nodes:
        for v in vertex_neighbors[u]:
            if u < v:
                skeletal_edges.append((u, v))

    return skeletal_nodes, skeletal_edges, parent


def refine_skeleton_embedding(
    original_vertices,
    contracted_vertices,
    skeletal_nodes,
    skeletal_edges,
    parent,
    original_faces,
):
    """
    Embedding Refinement (Section 6) using original boundary loops.
    """
    num_vertices = len(original_vertices)
    original_neighbors = {v: set() for v in range(num_vertices)}
    for f in original_faces:
        original_neighbors[f[0]].add(f[1])
        original_neighbors[f[0]].add(f[2])
        original_neighbors[f[1]].add(f[0])
        original_neighbors[f[1]].add(f[2])
        original_neighbors[f[2]].add(f[0])
        original_neighbors[f[2]].add(f[1])

    Pi = {node: [] for node in skeletal_nodes}
    for u in range(num_vertices):
        rep = find_node(u, parent)
        if rep in Pi:
            Pi[rep].append(u)

    node_degree = {node: 0 for node in skeletal_nodes}
    for u, v in skeletal_edges:
        node_degree[u] += 1
        node_degree[v] += 1

    refined_positions = {}
    for node in skeletal_nodes:
        region = set(Pi[node])
        if not region:
            refined_positions[node] = contracted_vertices[node].copy()
            continue

        deg = node_degree[node]
        if deg <= 1:
            # Terminal node: average displacement of region
            displacements = [
                contracted_vertices[i] - original_vertices[i] for i in region
            ]
            d = np.mean(displacements, axis=0)
            refined_positions[node] = contracted_vertices[node] - d
        else:
            # Regular or Junction node: boundary loop displacements
            boundary_verts = set()
            for u in region:
                for w in original_neighbors[u]:
                    if w not in region:
                        boundary_verts.add(u)
                        break

            if not boundary_verts:
                refined_positions[node] = np.mean(
                    original_vertices[list(region)], axis=0
                )
                continue

            # Connected components of boundary graph
            visited = set()
            loops = []
            adj = {u: [] for u in boundary_verts}
            for u in boundary_verts:
                for w in original_neighbors[u]:
                    if w in boundary_verts:
                        adj[u].append(w)

            for u in boundary_verts:
                if u not in visited:
                    comp = []
                    queue = [u]
                    visited.add(u)
                    while queue:
                        curr = queue.pop(0)
                        comp.append(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    loops.append(comp)

            d_list = []
            len_list = []
            for loop in loops:
                loop_set = set(loop)
                d_j_num = np.zeros(3)
                d_j_den = 0.0
                total_loop_len = 0.0

                for i in loop:
                    adj_in_loop = [w for w in adj[i] if w in loop_set]
                    l_ji = 0.0
                    for w in adj_in_loop:
                        edge_len = np.linalg.norm(
                            original_vertices[i] - original_vertices[w]
                        )
                        l_ji += edge_len
                        total_loop_len += edge_len

                    if l_ji < 1e-12:
                        l_ji = 1.0

                    disp_i = contracted_vertices[i] - original_vertices[i]
                    d_j_num += l_ji * disp_i
                    d_j_den += l_ji

                if d_j_den > 1e-12:
                    d_j = d_j_num / d_j_den
                else:
                    d_j = np.mean(
                        [contracted_vertices[i] - original_vertices[i] for i in loop],
                        axis=0,
                    )

                d_list.append(d_j)
                len_list.append(total_loop_len)

            if not d_list:
                refined_positions[node] = np.mean(
                    original_vertices[list(region)], axis=0
                )
                continue

            if deg == 2 and len(d_list) >= 2:
                shift = (d_list[0] + d_list[1]) / 2.0
                refined_positions[node] = contracted_vertices[node] - shift
            else:
                sum_len = sum(len_list)
                if sum_len > 1e-12:
                    shift = (
                        sum(length_val * d for length_val, d in zip(len_list, d_list))
                        / sum_len
                    )
                else:
                    shift = np.mean(d_list, axis=0)
                refined_positions[node] = contracted_vertices[node] - shift

    return refined_positions


def merge_junction_nodes(
    skeletal_nodes,
    skeletal_edges,
    parent,
    original_vertices,
    node_positions,
):
    """
    Junction Node Merging (Au et al. 2008, Section 6).
    Iteratively merges junction nodes with adjacent neighbors if the merged node
    has better centeredness: sigma'_k < 0.9 * sigma_k.
    """
    num_original = len(original_vertices)
    active_nodes = list(skeletal_nodes)
    current_edges = list(skeletal_edges)
    curr_parent = list(parent)
    pos_map = {node: node_positions[node].copy() for node in skeletal_nodes}

    def get_region(node):
        return [u for u in range(num_original) if find_node(u, curr_parent) == node]

    def calc_centeredness(node, pos):
        region = get_region(node)
        if not region:
            return 0.0
        dists = np.linalg.norm(original_vertices[region] - pos, axis=1)
        return float(np.std(dists))

    merged_any = True
    while merged_any:
        merged_any = False

        adj = {u: set() for u in active_nodes}
        for u, v in current_edges:
            if u in adj and v in adj and u != v:
                adj[u].add(v)
                adj[v].add(u)

        junction_nodes = [u for u in active_nodes if len(adj[u]) >= 3]

        for k in junction_nodes:
            if k not in active_nodes:
                continue

            sigma_k = calc_centeredness(k, pos_map[k])
            if sigma_k < 1e-12:
                continue

            best_neighbor = None
            best_sigma_prime = sigma_k

            for neighbor in adj[k]:
                region_k = get_region(k)
                region_m = get_region(neighbor)
                merged_region = region_k + region_m
                if not merged_region:
                    continue

                pos_merged = np.mean(original_vertices[merged_region], axis=0)
                dists_merged = np.linalg.norm(
                    original_vertices[merged_region] - pos_merged, axis=1
                )
                sigma_prime = float(np.std(dists_merged))

                if sigma_prime < 0.9 * sigma_k and sigma_prime < best_sigma_prime:
                    best_sigma_prime = sigma_prime
                    best_neighbor = neighbor

            if best_neighbor is not None:
                for u in get_region(k):
                    curr_parent[u] = best_neighbor

                active_nodes.remove(k)
                pos_map[best_neighbor] = np.mean(
                    original_vertices[get_region(best_neighbor)], axis=0
                )

                new_edges = []
                for u, v in current_edges:
                    u_rep = best_neighbor if u == k else u
                    v_rep = best_neighbor if v == k else v
                    if u_rep != v_rep:
                        edge = (min(u_rep, v_rep), max(u_rep, v_rep))
                        if edge not in new_edges:
                            new_edges.append(edge)
                current_edges = new_edges
                merged_any = True
                break

    sorted_nodes = sorted(list(active_nodes))
    final_positions = {node: pos_map[node] for node in sorted_nodes}

    return sorted_nodes, current_edges, curr_parent, final_positions


def extract_skeleton(
    mesh: trimesh.Trimesh,
    max_iters=20,
    epsilon=1e-6,
    threshold=0.5,
    no_1d_collapses=False,
    enable_embedding_refinement=True,
    enable_junction_merging=True,
    return_tuple=False,
    use_pytorch=False,
    device=None,
):
    """
    Extract a 1D curve skeleton from a 3D mesh using the full Au et al. (2008) pipeline.
    """
    # Step 1: Geometry Contraction
    contracted_vertices = contract_mesh(
        mesh.vertices,
        mesh.faces,
        max_iters=max_iters,
        epsilon=epsilon,
        use_pytorch=use_pytorch,
        device=device,
    )

    # Step 2: Connectivity Surgery
    skeletal_nodes, skeletal_edges, parent = connectivity_surgery(
        mesh.faces, contracted_vertices, threshold, no_1d_collapses
    )

    # Step 3: Embedding Refinement & Junction Merging
    if enable_embedding_refinement:
        node_positions = refine_skeleton_embedding(
            mesh.vertices,
            contracted_vertices,
            skeletal_nodes,
            skeletal_edges,
            parent,
            mesh.faces,
        )
    else:
        node_positions = {
            node: contracted_vertices[node].copy() for node in skeletal_nodes
        }

    if enable_junction_merging and enable_embedding_refinement:
        skeletal_nodes, skeletal_edges, parent, node_positions = merge_junction_nodes(
            skeletal_nodes,
            skeletal_edges,
            parent,
            mesh.vertices,
            node_positions,
        )

    node_to_idx = {node: idx for idx, node in enumerate(skeletal_nodes)}
    skeleton_vertices = np.array([node_positions[node] for node in skeletal_nodes])
    skeleton_edges = np.array(
        [[node_to_idx[u], node_to_idx[v]] for u, v in skeletal_edges]
    )

    if return_tuple:
        return skeleton_vertices, skeleton_edges

    return trimesh.load_path(skeleton_vertices[skeleton_edges])
