import torch
from jaxtyping import Float

Vec3 = tuple[float, float, float]
PoseTransformTensor = Float[torch.Tensor, "4 4"]
Vector3Tensor = Float[torch.Tensor, "3"]
RotationMatrixTensor = Float[torch.Tensor, "3 3"]
AnimationFrame = list[RotationMatrixTensor]
TimeSeconds = float
Animation = dict[TimeSeconds, AnimationFrame]
