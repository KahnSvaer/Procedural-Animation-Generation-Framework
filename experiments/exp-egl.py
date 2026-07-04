import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

print("Creating renderer...")
r = pyrender.OffscreenRenderer(64, 64)
print("Renderer created!")

scene = pyrender.Scene()
color, depth = r.render(scene)

print(color.shape)