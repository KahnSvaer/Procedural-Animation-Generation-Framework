import trimesh
from typing import List
import numpy as np
import pyrender

def _look_at(camera_position, target=[0, 0, 0], up=[0, 1, 0]):
    camera_position = np.array(camera_position, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)

    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position

    return pose


def _generate_poses(vertical_angles: np.ndarray, horizontal_angles: np.ndarray, distance: float):
    #MonkeyPatch to avoid numpy deprecation warning in pyrender
    np.infty = np.inf

    poses = []

    for angle in vertical_angles:

        horizontal_distance = distance * np.cos(angle)
        vertical_distance = distance * np.sin(angle)

        for horizontal_angle in horizontal_angles:
            x = horizontal_distance * np.cos(horizontal_angle)
            z = horizontal_distance * np.sin(horizontal_angle)
            y = vertical_distance

            camera_pos = [x, y, z]

            pose = _look_at(camera_pos)

            poses.append(pose)

    return np.array(poses)



def render_views(
    mesh: trimesh.Trimesh,
    num_views: int = 8,
    vertical_angles: List[float] = [-30, 0, 30],
    viewport_size: tuple = (640, 480),
) -> tuple[List[np.ndarray], np.ndarray]:

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.3, 0.3, 0.3]
    )

    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=viewport_size[0],
        viewport_height=viewport_size[1]
    )

    horizontal_angles = np.linspace(0, 360, num_views, endpoint=False)
    
    horizontal_angles_rad = np.radians(horizontal_angles)
    vertical_angles_rad = np.radians(vertical_angles)

    rendered_images = []

    camera_pose_list = _generate_poses(
        vertical_angles=vertical_angles_rad, 
        horizontal_angles=horizontal_angles_rad, 
        distance=2.0
    ) # pre-generate camera poses for debugging

    for camera_pose in camera_pose_list:

        camera = pyrender.PerspectiveCamera(
            yfov=np.pi / 3.0
        )

        light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=3.0
        )

        camera_node = scene.add(
            camera,
            pose=camera_pose
        )

        light_node = scene.add(
            light,
            pose=camera_pose
        )

        result = renderer.render(scene)
        if result is None:
            raise RuntimeError("Renderer returned None, rendering failed.")
        
        color, _ = result

        rendered_images.append(color)

        scene.remove_node(camera_node)
        scene.remove_node(light_node)

    renderer.delete()
    return rendered_images, camera_pose_list # camera poses returned for debugging 