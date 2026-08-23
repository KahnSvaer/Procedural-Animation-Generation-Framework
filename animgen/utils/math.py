import numpy as np
import torch


def range_norm(
    t: torch.Tensor | np.ndarray,
    lb: float | torch.Tensor | np.ndarray | None = None,
    ub: float | torch.Tensor | np.ndarray | None = None,
    offset: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor | np.ndarray:
    """
    Normalize continuous tensor or array values to the [0, 1] range.

    Parameters
    ----------
    t : torch.Tensor | np.ndarray
        Input tensor or array of continuous values to normalize.
    lb : float | torch.Tensor | np.ndarray | None, default=None
        Lower bound. If None, computed as min(t) (or min(t) - offset if offset is provided).
    ub : float | torch.Tensor | np.ndarray | None, default=None
        Upper bound. If None, computed as max(t).
    offset : float | None, default=None
        Optional offset subtracted from minimum value when lb is None.
    eps : float, default=1e-8
        Small epsilon value to prevent division by zero.

    Returns
    -------
    normalized : torch.Tensor | np.ndarray
        Range-normalized values scaled between 0 and 1.
    """
    if lb is None:
        lb = t.min() - offset if offset else t.min()
    if ub is None:
        ub = t.max()
    return (t - lb) / (ub - lb + eps)


def rotation_matrix_from_vectors(
    a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """
    Compute the 3x3 rotation matrix that aligns vector a to vector b.

    Parameters
    ----------
    a : (3,) torch.Tensor | np.ndarray
        Source 3D vector.
    b : (3,) torch.Tensor | np.ndarray
        Target 3D vector.

    Returns
    -------
    R : (3, 3) torch.Tensor | np.ndarray
        The 3x3 rotation matrix mapping a to b such that R @ a / ||a|| == b / ||b||.
        Preserves the input type (PyTorch Tensor or NumPy ndarray).
    """
    is_numpy = isinstance(a, np.ndarray) or isinstance(b, np.ndarray)

    a_t = torch.as_tensor(a, dtype=torch.float64)
    b_t = torch.as_tensor(b, dtype=torch.float64)

    a_t = a_t / torch.linalg.norm(a_t)
    b_t = b_t / torch.linalg.norm(b_t)
    v = torch.linalg.cross(a_t, b_t)
    c = torch.dot(a_t, b_t)
    s = torch.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            R = torch.eye(3, dtype=a_t.dtype, device=a_t.device)
        else:
            if abs(a_t[0].item()) < 0.9:
                ortho = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=a_t.dtype, device=a_t.device
                )
            else:
                ortho = torch.tensor(
                    [0.0, 1.0, 0.0], dtype=a_t.dtype, device=a_t.device
                )
            ortho = ortho - torch.dot(ortho, a_t) * a_t
            ortho = ortho / torch.linalg.norm(ortho)
            K = torch.stack(
                [
                    torch.stack(
                        [
                            torch.tensor(0.0, dtype=a_t.dtype, device=a_t.device),
                            -ortho[2],
                            ortho[1],
                        ]
                    ),
                    torch.stack(
                        [
                            ortho[2],
                            torch.tensor(0.0, dtype=a_t.dtype, device=a_t.device),
                            -ortho[0],
                        ]
                    ),
                    torch.stack(
                        [
                            -ortho[1],
                            ortho[0],
                            torch.tensor(0.0, dtype=a_t.dtype, device=a_t.device),
                        ]
                    ),
                ]
            )
            R = torch.eye(3, dtype=a_t.dtype, device=a_t.device) + 2.0 * torch.matmul(
                K, K
            )
    else:
        K = torch.stack(
            [
                torch.stack(
                    [torch.tensor(0.0, dtype=a_t.dtype, device=a_t.device), -v[2], v[1]]
                ),
                torch.stack(
                    [v[2], torch.tensor(0.0, dtype=a_t.dtype, device=a_t.device), -v[0]]
                ),
                torch.stack(
                    [-v[1], v[0], torch.tensor(0.0, dtype=a_t.dtype, device=a_t.device)]
                ),
            ]
        )
        R = (
            torch.eye(3, dtype=a_t.dtype, device=a_t.device)
            + K
            + torch.matmul(K, K) * ((1.0 - c) / (s**2))
        )

    if is_numpy:
        return R.cpu().numpy()
    return R


def rotation_matrix_to_quaternion(R: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix into a normalized unit quaternion [w, x, y, z].

    Uses Shepperd's algorithm for numerically stable conversion across all rotation angles.

    Parameters
    ----------
    R : (3, 3) np.ndarray | torch.Tensor
        The 3x3 orthogonal rotation matrix in SO(3).

    Returns
    -------
    q : (4,) np.ndarray
        Normalized unit quaternion representation formatted as [w, x, y, z] (scalar-first).
    """
    if isinstance(R, torch.Tensor):
        R_np = R.detach().cpu().numpy().astype(np.float64)
    else:
        R_np = np.asarray(R, dtype=np.float64)

    trace = float(R_np[0, 0] + R_np[1, 1] + R_np[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R_np[2, 1] - R_np[1, 2]) * s
        y = (R_np[0, 2] - R_np[2, 0]) * s
        z = (R_np[1, 0] - R_np[0, 1]) * s
    elif (R_np[0, 0] > R_np[1, 1]) and (R_np[0, 0] > R_np[2, 2]):
        s = 2.0 * np.sqrt(max(1.0 + R_np[0, 0] - R_np[1, 1] - R_np[2, 2], 1e-12))
        w = (R_np[2, 1] - R_np[1, 2]) / s
        x = 0.25 * s
        y = (R_np[0, 1] + R_np[1, 0]) / s
        z = (R_np[0, 2] + R_np[2, 0]) / s
    elif R_np[1, 1] > R_np[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R_np[1, 1] - R_np[0, 0] - R_np[2, 2], 1e-12))
        w = (R_np[0, 2] - R_np[2, 0]) / s
        x = (R_np[0, 1] + R_np[1, 0]) / s
        y = 0.25 * s
        z = (R_np[1, 2] + R_np[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + R_np[2, 2] - R_np[0, 0] - R_np[1, 1], 1e-12))
        w = (R_np[1, 0] - R_np[0, 1]) / s
        x = (R_np[0, 2] + R_np[2, 0]) / s
        y = (R_np[1, 2] + R_np[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q = q / norm
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q


def quaternion_to_rotation_matrix(q: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Parameters
    ----------
    q : (4,) np.ndarray | torch.Tensor
        Unit quaternion represented as [w, x, y, z] (scalar-first).

    Returns
    -------
    R : (3, 3) np.ndarray
        The corresponding 3x3 orthogonal rotation matrix in SO(3).
    """
    if isinstance(q, torch.Tensor):
        q_np = q.detach().cpu().numpy().astype(np.float64)
    else:
        q_np = np.asarray(q, dtype=np.float64)

    norm = np.linalg.norm(q_np)
    if norm > 1e-12:
        q_np = q_np / norm
    else:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q_np
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_slerp(
    q1: np.ndarray | torch.Tensor,
    q2: np.ndarray | torch.Tensor,
    alpha: float,
) -> np.ndarray:
    """
    Perform Spherical Linear Interpolation (SLERP) between two unit quaternions.

    Interpolates along the shortest geodesic arc on the 4D unit hypersphere.

    Parameters
    ----------
    q1 : (4,) np.ndarray | torch.Tensor
        Initial unit quaternion at alpha = 0.0, formatted as [w, x, y, z].
    q2 : (4,) np.ndarray | torch.Tensor
        Final unit quaternion at alpha = 1.0, formatted as [w, x, y, z].
    alpha : float
        Interpolation weight in [0.0, 1.0].

    Returns
    -------
    q : (4,) np.ndarray
        Interpolated unit quaternion formatted as [w, x, y, z].
    """
    if isinstance(q1, torch.Tensor):
        q1_np = q1.detach().cpu().numpy().astype(np.float64)
    else:
        q1_np = np.asarray(q1, dtype=np.float64).copy()

    if isinstance(q2, torch.Tensor):
        q2_np = q2.detach().cpu().numpy().astype(np.float64)
    else:
        q2_np = np.asarray(q2, dtype=np.float64).copy()

    dot = float(np.dot(q1_np, q2_np))
    if dot < 0.0:
        q2_np = -q2_np
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q1_np + alpha * (q2_np - q1_np)
        norm = np.linalg.norm(result)
        return result / (norm if norm > 1e-12 else 1.0)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return s1 * q1_np + s2 * q2_np


def slerp_rotation_matrix(
    R1: torch.Tensor | np.ndarray,
    R2: torch.Tensor | np.ndarray,
    alpha: float,
) -> torch.Tensor:
    """
    Perform Spherical Linear Interpolation (SLERP) between two 3x3 rotation matrices.

    Converts rotation matrices to quaternions, evaluates SLERP along the shortest arc,
    and converts the interpolated result back to a 3x3 rotation matrix.

    Parameters
    ----------
    R1 : (3, 3) torch.Tensor | np.ndarray
        Initial 3x3 rotation matrix at alpha = 0.0.
    R2 : (3, 3) torch.Tensor | np.ndarray
        Final 3x3 rotation matrix at alpha = 1.0.
    alpha : float
        Interpolation parameter in [0.0, 1.0].

    Returns
    -------
    R : (3, 3) torch.Tensor
        Interpolated 3x3 rotation matrix as a PyTorch FloatTensor.
    """
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)
    q_interp = quaternion_slerp(q1, q2, alpha)
    R_np = quaternion_to_rotation_matrix(q_interp)
    return torch.tensor(R_np, dtype=torch.float32)


def quaternion_multiply(
    q1: np.ndarray | torch.Tensor,
    q2: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """
    Compute the Hamilton product of two quaternions q1 and q2.

    Parameters
    ----------
    q1 : (4,) np.ndarray | torch.Tensor
        First quaternion represented as [w, x, y, z].
    q2 : (4,) np.ndarray | torch.Tensor
        Second quaternion represented as [w, x, y, z].

    Returns
    -------
    q : (4,) np.ndarray
        Hamilton product q = q1 * q2 represented as [w, x, y, z].
    """
    if isinstance(q1, torch.Tensor):
        q1_np = q1.detach().cpu().numpy().astype(np.float64)
    else:
        q1_np = np.asarray(q1, dtype=np.float64)

    if isinstance(q2, torch.Tensor):
        q2_np = q2.detach().cpu().numpy().astype(np.float64)
    else:
        q2_np = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1_np
    w2, x2, y2, z2 = q2_np

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_conjugate(q: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Compute the conjugate of a quaternion [w, x, y, z] -> [w, -x, -y, -z].

    For unit quaternions, the conjugate represents the inverse rotation.

    Parameters
    ----------
    q : (4,) np.ndarray | torch.Tensor
        Quaternion represented as [w, x, y, z].

    Returns
    -------
    q_conj : (4,) np.ndarray
        Conjugated quaternion [w, -x, -y, -z].
    """
    if isinstance(q, torch.Tensor):
        q_np = q.detach().cpu().numpy().astype(np.float64)
    else:
        q_np = np.asarray(q, dtype=np.float64)
    return np.array([q_np[0], -q_np[1], -q_np[2], -q_np[3]], dtype=np.float64)
