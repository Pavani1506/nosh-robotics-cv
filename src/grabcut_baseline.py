from pathlib import Path
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
OUT = Path("outputs/classical")
OUT.mkdir(parents=True, exist_ok=True)

images = sorted(
    p for p in (ROOT / "train" / "egg").rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)
if not images:
    raise SystemExit("No training images found.")

image_path = images[0]
image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit(f"Could not read {image_path}")

h, w = image.shape[:2]

# Initial experiment: rectangle around the central egg.
# This assumes the egg is near the center; inspect before generalizing.
rect = (
    int(w * 0.27),
    int(h * 0.25),
    int(w * 0.56),
    int(h * 0.58),
)

mask = np.zeros((h, w), np.uint8)
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

cv2.grabCut(
    image,
    mask,
    rect,
    bg_model,
    fg_model,
    5,
    cv2.GC_INIT_WITH_RECT,
)

result_mask = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    255, 0
).astype(np.uint8)

cv2.imwrite(str(OUT / "trial3_grabcut_mask.png"), result_mask)

overlay = image.copy()
green = np.zeros_like(image)
green[:] = (0, 255, 0)
selected = result_mask > 0
overlay[selected] = (
    0.55 * image[selected] + 0.45 * green[selected]
).astype(np.uint8)

cv2.imwrite(str(OUT / "trial3_grabcut_overlay.jpg"), overlay)

print("Image:", image_path)
print("Saved:", OUT / "trial3_grabcut_mask.png")
print("Saved:", OUT / "trial3_grabcut_overlay.jpg")
