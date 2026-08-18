from animgen.core.types import AnimationFrame
import numpy as np
import torch


def range_norm(t, lb=None, ub=None, offset=None, eps=1e-8):
    """
    Given tensor of continuous values, return corresponding range normalized values.
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
    Compute the 3x3 rotation matrix that rotates vector a to vector b.
    Supports both NumPy arrays and PyTorch Tensors.
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


def successive_rotations(
    src: torch.Tensor | np.ndarray,
    tgt: torch.Tensor | np.ndarray,
    is_positions: bool = False,
) -> AnimationFrame:
    """
    Compute successive rotation matrices for a hierarchical chain of segments,
    accounting for the accumulated rotation of parent segments.

    Parameters
    ----------
    src : (N, 3) or (N+1, 3) Tensor or ndarray
        The source segment vectors or joint positions.
    tgt : (N, 3) or (N+1, 3) Tensor or ndarray
        The target segment vectors or joint positions.
    is_positions : bool, default=False
        If True, the inputs are treated as joint positions and are diffed along the first dimension
        to produce segment vectors.

    Returns
    -------
    rotations : AnimationFrame
        The local rotation matrices for each segment.
    """
    src_t = torch.as_tensor(src, dtype=torch.float64)
    tgt_t = torch.as_tensor(tgt, dtype=torch.float64)

    if is_positions:
        src_t = torch.diff(src_t, dim=0)
        tgt_t = torch.diff(tgt_t, dim=0)

    N = len(src_t)
    rotations = []
    R_accum = torch.eye(3, dtype=torch.float64, device=src_t.device)

    for i in range(N):
        src_rotated = R_accum @ src_t[i]
        R_local = rotation_matrix_from_vectors(src_rotated, tgt_t[i])
        rotations.append(R_local)
        R_accum = R_local @ R_accum

    return rotations
