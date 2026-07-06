"""
Reference - "https://huggingface.co/facebook/sam3"
"""

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Env loading
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = "./models_cache/SAM_original/"

MODEL_PATH = "facebook/sam3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILE_PATH = Path(__file__).parent

# MODEL SETUP
model = Sam3Model.from_pretrained(MODEL_PATH).to(DEVICE)
processor = Sam3Processor.from_pretrained(MODEL_PATH)

image_path = FILE_PATH / "test_image.png"
image = Image.open(image_path).convert("RGB")

#
inputs = processor(images=image, text="dorsal fins", return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]

print(f"Found {len(results['masks'])} objects")


mask = results["masks"][0].detach().cpu()
mask_np = mask.numpy()
mask_vis = (mask_np * 255).astype(np.uint8)

mask_pil = Image.fromarray(mask_vis)

original_np = np.array(image)

overlay = original_np.copy()

overlay[mask_np > 0] = (
    overlay[mask_np > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
).astype(np.uint8)

overlay_pil = Image.fromarray(overlay)

combined = Image.new("RGB", (image.width * 3, image.height))

combined.paste(image, (0, 0))
combined.paste(mask_pil.convert("RGB"), (image.width, 0))
combined.paste(overlay_pil, (image.width * 2, 0))

combined.save(FILE_PATH / "SAM_original_comparison.png")
combined.show()
