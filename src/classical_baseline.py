from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
OUT = Path("outputs/classical")
OUT.mkdir(parents=True, exist_ok=True)

# Start with one image from the training egg folder.
candidates = sorted((ROOT / "train" / "egg").glob("*"))
image_paths = [
    p for p in candidates
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
]

if not image_paths:
    raise SystemExit(f"No training images found in {ROOT / 'train' / 'egg'}")

image_path = image_paths[0]
image = cv2.imread(str(image_path))

if image is None:
    raise SystemExit(f"Could not read image: {image_path}")

# Trial 1: Otsu thresholding on grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, otsu_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Trial 2: threshold bright, low-saturation pixels in HSV.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)

color_mask = cv2.inRange(
    hsv,
    np.array([0, 0, 125], dtype=np.uint8),
    np.array([179, 110, 255], dtype=np.uint8),
)

# Clean small noise and smooth boundaries.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(
    color_mask, cv2.MORPH_OPEN, kernel, iterations=1
)
clean_mask = cv2.morphologyEx(
    clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2
)

# Save the masks independently for visual comparison.
cv2.imwrite(str(OUT / "trial1_otsu.png"), otsu_mask)
cv2.imwrite(str(OUT / "trial2_hsv.png"), color_mask)
cv2.imwrite(str(OUT / "trial2_hsv_clean.png"), clean_mask)

# Overlay the cleaned mask in green on the original image.
overlay = image.copy()
green = np.zeros_like(image)
green[:, :] = (0, 255, 0)
selected = clean_mask > 0
overlay[selected] = (
    0.55 * image[selected] + 0.45 * green[selected]
).astype(np.uint8)

cv2.imwrite(str(OUT / "trial2_overlay.jpg"), overlay)

print("Image:", image_path)
print("Image dimensions:", image.shape[1], "x", image.shape[0])
print("Trial 1: Otsu threshold saved.")
print("Trial 2: HSV threshold and cleaned mask saved.")
print("Outputs:", OUT.resolve())
print("Note: These are candidate foreground masks, not verified egg-only masks.")
