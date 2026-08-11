import numpy as np


def range_norm(t, lb=None, ub=None, offset=None, eps=1e-8):
    """
    Given tensor of continuous values, return corresponding range normalized values.
    """
    if lb is None:
        lb = t.min() - offset if offset else t.min()
    if ub is None:
        ub = t.max()
    return (t - lb) / (ub - lb + eps)


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix that rotates vector a to vector b.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            if abs(a[0]) < 0.9:
                ortho = np.array([1.0, 0.0, 0.0])
            else:
                ortho = np.array([0.0, 1.0, 0.0])
            ortho = ortho - np.dot(ortho, a) * a
            ortho = ortho / np.linalg.norm(ortho)
            K = np.array(
                [
                    [0, -ortho[2], ortho[1]],
                    [ortho[2], 0, -ortho[0]],
                    [-ortho[1], ortho[0], 0],
                ]
            )
            return np.eye(3) + 2.0 * np.dot(K, K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + np.dot(K, K) * ((1.0 - c) / (s**2))
    return R
