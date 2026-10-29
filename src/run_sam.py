from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

IMAGE_PATH = Path("outputs/first_image.jpg")
CHECKPOINT = Path("models/sam_vit_b_01ec64.pth")
OUTPUT_DIR = Path("outputs/sam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

image_bgr = cv2.imread(str(IMAGE_PATH))
if image_bgr is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("Loading SAM ViT-B...")
sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.90,
    crop_n_layers=0,
    min_mask_region_area=100,
)

# Fix float64 coordinates for Apple's MPS backend
mask_generator.point_grids = [
    grid.astype(np.float32)
    for grid in mask_generator.point_grids
]

print("Generating masks...")
masks = mask_generator.generate(image_rgb)
print("Masks generated:", len(masks))

overlay = image_rgb.copy()
for mask_data in masks:
    mask = mask_data["segmentation"]
    color = np.random.default_rng(
        int(mask_data["area"]) % 10000
    ).integers(50, 256, size=3, dtype=np.uint8)
    overlay[mask] = (
        0.55 * overlay[mask] + 0.45 * color
    ).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title(f"SAM masks: {len(masks)}")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "first_image_masks.png", dpi=150)
plt.close()

print("Saved:", OUTPUT_DIR / "first_image_masks.png")
