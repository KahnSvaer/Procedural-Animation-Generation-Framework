import sys
sys.path.append(".") # Add animgen to path

from PIL import Image
import numpy as np
from pathlib import Path

from animgen.renderer.renderer import Renderer, render_multiview, colormap_faces, colormap_norms
from animgen.appendage_discovery.shape_diameter_function import (
    shape_diameter_function,
    colormap_shape_diameter_function,
    prep_mesh_shape_diameter_function,
)
from animgen.io.model_io import load_model

SAMPLE_OBJECT = "./generated_data/models/paint_mesh_Dolphin.glb"

if __name__ == '__main__':
    source2 = load_model(SAMPLE_OBJECT)
    #source2 = remove_texture(source2, visual_kind='vertex')
    source2 = prep_mesh_shape_diameter_function(source2)
    source2 = colormap_shape_diameter_function(source2, shape_diameter_function(source2))

    pose = np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  2.5],
        [ 0,  0,  0,  1],
    ])

    renderer = Renderer()
    renderer.set_object(source2, smooth=False)
    renderer.set_camera()
    renders = renderer.render(pose, interpolate_norms=True)
    for k, v in renders.items():
        print(k, v.shape)
    image = Image.fromarray(renders['matte'])
    image.save('test_matte_scene.png')
    image_faceids = colormap_faces(renders['faces'])
    image_faceids.save('test_faceids_scene.png')
    image_norms = colormap_norms(renders['norms'])
