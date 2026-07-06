import numpy as np

import numpy as np
import torch
import torch.nn.functional as F

from animgen.core.types import Vec3Tensor, PoseTransforms
from animgen.utils.camera_position import POSITION_GENERATORS


def matrix3x4_to_4x4(matrix3x4) -> PoseTransforms:
    """
    Convert a 3x4 transformation matrix to a 4x4 transformation matrix.
    """
    bottom = torch.zeros_like(matrix3x4[:, 0, :].unsqueeze(-2))
    bottom[..., -1] = 1
    return torch.cat([matrix3x4, bottom], dim=-2)


def view_matrix(
    camera_positions: Vec3Tensor,
    lookat_position: Vec3Tensor = torch.tensor([0, 0, 0]),
    up: Vec3Tensor = torch.tensor([0, 1, 0]),
) -> PoseTransforms:
    """
    Given lookat position, camera position, and up vector, compute cam2world poses.
    """
    if camera_positions.ndim == 1:
        camera_positions = camera_positions.unsqueeze(0)
    if lookat_position.ndim == 1:
        lookat_position = lookat_position.unsqueeze(0)
    camera_positions = camera_positions.float()
    lookat_position = lookat_position.float()

    cam_u = (
        up.unsqueeze(0)
        .repeat(len(lookat_position), 1)
        .float()
        .to(camera_positions.device)
    )

    # handle degenerate cases
    crossp = (
        torch.abs(torch.cross(lookat_position - camera_positions, cam_u, dim=-1))
        .max(dim=-1)
        .values
    )
    camera_positions[crossp < 1e-6] += 1e-6

    cam_z = F.normalize((lookat_position - camera_positions), dim=-1)
    cam_x = F.normalize(torch.cross(cam_z, cam_u, dim=-1), dim=-1)
    cam_y = F.normalize(torch.cross(cam_x, cam_z, dim=-1), dim=-1)
    poses = torch.stack(
        [cam_x, cam_y, -cam_z, camera_positions], dim=-1
    )  # same as nerfstudio convention [right, up, -lookat]
    poses = matrix3x4_to_4x4(poses)
    return poses


def sample_view_matrices(
    n: int, radius: float, lookat_position: Vec3Tensor = torch.tensor([0, 0, 0])
) -> PoseTransforms:
    """
    Sample n uniformly distributed view matrices spherically with given radius.
    """
    tht = torch.rand(n) * torch.pi * 2
    phi = torch.rand(n) * torch.pi
    world_x = radius * torch.sin(phi) * torch.cos(tht)
    world_y = radius * torch.sin(phi) * torch.sin(tht)
    world_z = radius * torch.cos(phi)
    camera_position = torch.stack([world_x, world_y, world_z], dim=-1)
    lookat_position = lookat_position.unsqueeze(0).repeat(n, 1)
    return view_matrix(
        camera_position.to(lookat_position.device),
        lookat_position,
        up=torch.tensor([0, 1, 0], device=lookat_position.device),
    )


def sample_view_matrices_polyhedra(
    polygon: str,
    radius: float,
    lookat_position: Vec3Tensor = torch.tensor([0, 0, 0]),
    **kwargs,
) -> PoseTransforms:
    """
    Sample view matrices according to a polygon with given radius.
    """
    polygon = polygon.strip().lower()
    if polygon not in POSITION_GENERATORS.keys():
        raise ValueError("Unknown shape chosen")
    camera_position = torch.from_numpy(POSITION_GENERATORS[polygon](**kwargs)) * radius
    return view_matrix(
        camera_position.to(lookat_position.device) + lookat_position,
        lookat_position,
        up=torch.tensor([0, 1, 0], device=lookat_position.device),
    )


if __name__ == "__main__":
    m = view_matrix(
        torch.tensor([0, 0, 1]).unsqueeze(0),
        torch.tensor([0, 0, 0]).unsqueeze(0),
    )
    print(m)

    m = view_matrix(
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 0, 0]),
    )
    print(m)

    for m in sample_view_matrices(10, 1):
        print(m)
