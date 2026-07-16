import torch
from jaxtyping import Float

Vec3 = tuple[float, float, float]
PoseTransforms = Float[torch.Tensor, "4 4"]
Vector3Tensor = Float[torch.Tensor, "3"]
