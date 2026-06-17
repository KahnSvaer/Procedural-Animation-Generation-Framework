import torch
import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    Materials,
    look_at_view_transform
)

from PIL import Image
import numpy as np

device = torch.device("cuda:0")
tm_mesh = trimesh.load_mesh("./generated_data/models/img_mesh_Crab.glb")

verts = torch.tensor(
    tm_mesh.vertices,
    dtype=torch.float32,
    device=device
)

faces = torch.tensor(
    tm_mesh.faces,
    dtype=torch.int64,
    device=device
)

verts_rgb = torch.ones(
    (1, verts.shape[0], 3),
    dtype=torch.float32,
    device=device
)

textures = TexturesVertex(
    verts_features=verts_rgb
)

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

R, T = look_at_view_transform(
    dist=3.0,   # distance from object
    elev=20,    # up/down angle
    azim=45     # left/right angle
)

cameras = FoVPerspectiveCameras(
    device=device,
    R=R,
    T=T
)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

lights = PointLights(
    device=device,
    location=T
)

materials = Materials(
    device=device,
    specular_color=((0.0, 0.0, 0.0),),
    shininess=0,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materials,
    )
)

image = renderer(mesh)
img = image[0, ..., :3]           # remove alpha channel
img = img.detach().cpu().numpy()  # tensor -> numpy
img = (img * 255).astype(np.uint8)
Image.fromarray(img).show()