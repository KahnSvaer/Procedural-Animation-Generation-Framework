import torch

from animgen.core.types import Vec3, Vector3Tensor


def tensor_to_vec3(tensor: Vector3Tensor) -> Vec3:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor.")

    if tensor.shape != (3,):
        raise ValueError(f"Expected tensor with shape (3,), got {tensor.shape}.")

    return (
        tensor[0].item(),
        tensor[1].item(),
        tensor[2].item(),
    )


def vec3_to_tensor(vec: Vec3) -> Vector3Tensor:
    if not isinstance(vec, tuple):
        raise TypeError("Expected a tuple.")

    if len(vec) != 3:
        raise ValueError(f"Expected tuple with 3 elements, got {len(vec)}.")

    if not all(isinstance(value, (int, float)) for value in vec):
        raise TypeError("All Vec3 elements must be int or float.")

    return torch.tensor(
        vec,
        dtype=torch.float32,
    )
