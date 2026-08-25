"""
Mesh geometry processing utilities.

Contains discrete differential geometry operators (Cotangent Laplace-Beltrami,
vertex mass matrices, Taubin smoothing) and geometric query functions.

References
----------
.. [1] Pinkall, U., & Polthier, K. (1993). Computing discrete minimal surfaces
       and their conjugates. Experimental Mathematics, 2(1), 15-36.
       https://doi.org/10.1080/10586458.1993.10504266
.. [2] Taubin, G. (1995). A signal processing approach to fair surface design.
       Proceedings of the 22nd annual conference on Computer graphics and interactive techniques, 351-358.
"""

from typing import Optional, Union, overload
import numpy as np
import scipy.sparse as sp
import trimesh


def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Centers a mesh at the origin by subtracting its vertex centroid.
    """
    mesh = mesh.copy()
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center
    return mesh


def duplicate_verts(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Duplicates vertices per face so each triangle has unique vertices.
    Useful before coloring mesh to avoid face interpolation in OpenGL.
    """
    verts = mesh.vertices[mesh.faces.reshape(-1), :]
    faces = np.arange(0, verts.shape[0])
    faces = faces.reshape(-1, 3)
    try:
        face_colors = mesh.visual.face_colors
    except (AttributeError, ValueError, IndexError):
        face_colors = np.full(
            (len(mesh.faces), 4), [200, 200, 200, 255], dtype=np.uint8
        )
    return trimesh.Trimesh(
        vertices=verts, faces=faces, face_colors=face_colors, process=False
    )


def taubin_smoothing(
    mesh: trimesh.Trimesh,
    lamb: float = 0.5,
    nu: float = 0.53,
    iterations: int = 10,
) -> trimesh.Trimesh:
    """
    Smooth a mesh using Taubin filtering (Laplacian smoothing with shrinkage compensation).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to be smoothed.
    lamb : float
        The shrinkage factor (0.0 < lamb < 1.0).
    nu : float
        The dilation factor (0.0 < nu < 1.0, typically nu > lamb).
    iterations : int
        The number of smoothing iterations.

    Returns
    -------
    smoothed_mesh : trimesh.Trimesh
        A smoothed copy of the input mesh.
    """
    mesh_copy = mesh.copy()
    trimesh.smoothing.filter_taubin(mesh_copy, lamb=lamb, nu=nu, iterations=iterations)
    return mesh_copy


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Computes the surface area of each triangle in the mesh.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices.

    Returns
    -------
    areas : (M,) ndarray
        Per-triangle surface area.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Computes the lumped Voronoi / barycentric area associated with each vertex
    (1/3 of incident triangle areas).

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices.

    Returns
    -------
    areas : (N,) ndarray
        Per-vertex surface area.
    """
    face_a = triangle_areas(vertices, faces)
    v_areas = np.zeros(len(vertices), dtype=np.float64)
    for i in range(3):
        np.add.at(v_areas, faces[:, i], face_a / 3.0)
    return v_areas


def cotangent(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes cot(theta) between vectors u and v: (u . v) / ||u x v||.

    Parameters
    ----------
    u : ndarray
        First vector or array of vectors of shape (..., 3).
    v : ndarray
        Second vector or array of vectors of shape (..., 3).

    Returns
    -------
    cot_val : ndarray
        Cotangent values clamped to [-1e4, 1e4] for numerical stability.
    """
    cross = np.cross(u, v)
    cross_norm = np.linalg.norm(cross, axis=-1)
    dot = np.sum(u * v, axis=-1)
    cot_val = dot / np.maximum(cross_norm, 1e-12)
    return np.clip(cot_val, -1e4, 1e4)


def dist_point_to_segment_vectorized(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Euclidean distances and projection points from an array of 3D points
    to a line segment defined by endpoints seg_a and seg_b.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) representing 3D query points.
    seg_a : np.ndarray
        Array of shape (3,) representing the start of the line segment.
    seg_b : np.ndarray
        Array of shape (3,) representing the end of the line segment.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - dists : 1D array of shape (N,) containing Euclidean distances to the segment.
        - proj : 2D array of shape (N, 3) containing the projected closest points on the segment.
    """
    ab = seg_b - seg_a
    ab_len_sq = float(np.dot(ab, ab))

    if ab_len_sq < 1e-12:
        diff = points - seg_a
        dists = np.linalg.norm(diff, axis=1)
        proj = np.tile(seg_a, (len(points), 1))
        return dists, proj

    t = np.clip(np.sum((points - seg_a) * ab, axis=1) / ab_len_sq, 0.0, 1.0)
    proj = seg_a + t[:, None] * ab[None, :]
    dists = np.linalg.norm(points - proj, axis=1)
    return dists, proj


@overload
def compute_cotangent_laplacian(
    mesh_or_vertices: trimesh.Trimesh,
    faces: None = None,
    return_mass_matrix: bool = False,
) -> Union[sp.csr_matrix, tuple[sp.csr_matrix, sp.csr_matrix]]: ...


@overload
def compute_cotangent_laplacian(
    mesh_or_vertices: np.ndarray,
    faces: np.ndarray,
    return_mass_matrix: bool = False,
) -> Union[sp.csr_matrix, tuple[sp.csr_matrix, sp.csr_matrix]]: ...


def compute_cotangent_laplacian(
    mesh_or_vertices: Union[trimesh.Trimesh, np.ndarray],
    faces: Optional[np.ndarray] = None,
    return_mass_matrix: bool = False,
) -> Union[sp.csr_matrix, tuple[sp.csr_matrix, sp.csr_matrix]]:
    """
    Computes the discrete cotangent Laplace-Beltrami operator L and optional
    lumped diagonal Mass matrix M for a triangular mesh with numerical regularization
    against degeneracies, non-manifold edges, and sliver faces.

    Parameters
    ----------
    mesh_or_vertices : trimesh.Trimesh | np.ndarray
        Either a trimesh.Trimesh instance or vertex coordinates of shape (N, 3).
    faces : np.ndarray, optional
        Triangle face indices of shape (M, 3) when vertices are passed directly.
    return_mass_matrix : bool, default=False
        If True, returns (L, M) tuple. If False, returns L.

    Returns
    -------
    sp.csr_matrix | tuple[sp.csr_matrix, sp.csr_matrix]
        - L : (N, N) symmetric positive semi-definite cotangent Laplacian matrix.
        - M : (N, N) diagonal lumped mass matrix (only if return_mass_matrix=True).
    """
    if isinstance(mesh_or_vertices, trimesh.Trimesh):
        V = np.asarray(mesh_or_vertices.vertices, dtype=np.float64)
        F = np.asarray(mesh_or_vertices.faces, dtype=np.int64)
    else:
        V = np.asarray(mesh_or_vertices, dtype=np.float64)
        if faces is None:
            raise ValueError("faces must be provided when vertices array is passed.")
        F = np.asarray(faces, dtype=np.int64)

    N = len(V)
    if len(F) == 0:
        L_empty = sp.csr_matrix((N, N), dtype=np.float64)
        if return_mass_matrix:
            return L_empty, sp.eye(N, dtype=np.float64, format="csr")
        return L_empty

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    cross = np.cross(e2, -e1)
    areas2 = np.linalg.norm(cross, axis=1)
    areas2_safe = np.maximum(areas2, 1e-12)

    cot0 = np.sum(e2 * (-e1), axis=1) / areas2_safe
    cot1 = np.sum(e0 * (-e2), axis=1) / areas2_safe
    cot2 = np.sum(e1 * (-e0), axis=1) / areas2_safe

    # Zero out cotangents for degenerate triangles
    degen_mask = areas2 < 1e-10
    if np.any(degen_mask):
        cot0[degen_mask] = 0.0
        cot1[degen_mask] = 0.0
        cot2[degen_mask] = 0.0

    # Clamp extreme cotangents for sliver triangles
    cot0 = np.clip(cot0, -1e4, 1e4)
    cot1 = np.clip(cot1, -1e4, 1e4)
    cot2 = np.clip(cot2, -1e4, 1e4)

    row_indices = np.concatenate([F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 0], F[:, 1]])
    col_indices = np.concatenate([F[:, 2], F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0]])
    W = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) * 0.5

    W_sp = sp.coo_matrix((-W, (row_indices, col_indices)), shape=(N, N)).tocsr()
    diag = -np.array(W_sp.sum(axis=1)).flatten()
    L = W_sp + sp.diags(diag, 0, shape=(N, N), format="csr")

    if not return_mass_matrix:
        return L

    # Lumped mass matrix (1/3 of incident triangle areas for each vertex)
    tri_areas = areas2 * 0.5
    M_diag = np.zeros(N, dtype=np.float64)
    for i in range(3):
        np.add.at(M_diag, F[:, i], tri_areas / 3.0)

    mean_area = float(np.mean(M_diag[M_diag > 0])) if np.any(M_diag > 0) else 1.0
    M_diag = np.maximum(M_diag, mean_area * 1e-6)
    M = sp.diags(M_diag, 0, shape=(N, N), format="csr")

    return L, M
