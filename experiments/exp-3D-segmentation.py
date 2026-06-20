"""
Reference - "https://huggingface.co/facebook/sam3"
"""

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

# Env loading
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ['HF_HOME'] = './models_cache/SAM_original/'

import sys
sys.path.append(".")
from animgen.core.generated_asset_class import GeneratedAssetClass

# Generating View
mesh_path = ("./generated_data/models/img_mesh_Goldfish.glb")
mesh = GeneratedAssetClass(mesh_path)

mesh_views = mesh.views
mesh_depths = mesh.depths

print(len(mesh_views), type(mesh_views[0]), mesh_views[0].shape, mesh_views[0].max())
print(len(mesh_depths), type(mesh_depths[0]), mesh_depths[0].shape, mesh_depths[0].max())

# Model Loading
MODEL_PATH = "facebook/sam3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL SETUP  
model = Sam3Model.from_pretrained(MODEL_PATH).to(DEVICE)
processor = Sam3Processor.from_pretrained(MODEL_PATH)

inputs = processor(images=mesh_views[4], text="dorsal fins", return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")


mask = results["masks"][0].detach().cpu()
mask_np = mask.numpy()
mask_vis = (mask_np * 255).astype(np.uint8)

mask_pil = Image.fromarray(mask_vis)
mask_pil.show()

