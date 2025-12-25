import torch
import numpy as np
from PIL import Image

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ------------------------------------------------------------------------------
# Utility: convert PIL image crop to visual prompt
# ------------------------------------------------------------------------------
def make_exemplar_prompt(crop):
    """
    Takes a PIL crop (example ferry)
    and returns a prompt dict for SAM3 (exemplar-based prompt).
    """
    # Convert to numpy and normalize
    arr = np.array(crop)
    # SAM3 uses visual exemplar prompt format
    return {"image_exemplars": [arr]}


# ------------------------------------------------------------------------------
# 1) Load model + processor
# ------------------------------------------------------------------------------
model = build_sam3_image_model().cuda()
processor = Sam3Processor(model)

# ------------------------------------------------------------------------------
# 2) Set target image
# ------------------------------------------------------------------------------
image = Image.open("ships_scene.jpg").convert("RGB")
state = processor.set_image(image)

# ------------------------------------------------------------------------------
# 3) Build prompts
# ------------------------------------------------------------------------------
# Text prompt: segment all ships
text_prompt = "ships"

# Visual exemplar prompt: ferry reference image crop
ferry_crop = Image.open("ferry_ref.jpg").convert("RGB")
visual_prompt = make_exemplar_prompt(ferry_crop)

# ------------------------------------------------------------------------------
# 4) Send combined prompt
# ------------------------------------------------------------------------------
response = processor.set_hybrid_prompts(
    state=state,
    prompt=text_prompt,
    **visual_prompt
)

# ------------------------------------------------------------------------------
# 5) Parse outputs
# ------------------------------------------------------------------------------
masks  = response["masks"]   # list of binary masks
boxes  = response["boxes"]   # bounding boxes
scores = response["scores"]  # confidence

print("Found", len(masks), "instances.")

# ------------------------------------------------------------------------------
# 6) Save segmented masks with labels
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

for idx, mask in enumerate(masks):
    plt.figure()
    plt.imshow(image)
    # mask is HxW uint8 — overlay with transparency
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Instance {idx}: ships/ferry segmentation")
    plt.axis("off")
    plt.savefig(f"seg_instance_{idx}.png")
