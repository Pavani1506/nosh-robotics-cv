from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
OUT = Path("outputs/classical_batch")
MASK_DIR = OUT / "masks"
OVERLAY_DIR = OUT / "overlays"

for folder in (MASK_DIR, OVERLAY_DIR):
    folder.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted(
    p for p in (ROOT / "train" / "egg").rglob("*")
    if p.suffix.lower() in extensions
)[:10]

if not images:
    raise SystemExit("No training images found.")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
results = []

def save_result(image_path, image, method, mask):
    relative = image_path.relative_to(ROOT)
    stem = relative.stem
    mask_path = MASK_DIR / method / relative.parent / f"{stem}.png"
    overlay_path = OVERLAY_DIR / method / relative.parent / f"{stem}.jpg"

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(mask_path), mask)

    overlay = image.copy()
    selected = mask > 0
    overlay[selected] = (
        0.55 * image[selected]
        + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    cv2.imwrite(str(overlay_path), overlay)

    results.append({
        "image": str(relative),
        "method": method,
        "parameters": {
            "otsu": "grayscale; binary Otsu",
            "hsv": "HSV lower=[0,0,125]; upper=[179,110,255]; ellipse 5x5; open=1; close=2",
            "grabcut": "central rectangle=(0.27w,0.25h,0.56w,0.58h); iterations=5",
        }[method],
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    })

for index, image_path in enumerate(images, start=1):
    image = cv2.imread(str(image_path))
    if image is None:
        print("Skipping unreadable image:", image_path)
        continue

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Otsu thresholding.
    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    save_result(image_path, image, "otsu", otsu)

    # Method 2: HSV thresholding and cleanup.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 125], dtype=np.uint8),
        np.array([179, 110, 255], dtype=np.uint8),
    )
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    save_result(image_path, image, "hsv", hsv_mask)

    # Method 3: GrabCut using a central rectangle.
    rect = (
        int(width * 0.27),
        int(height * 0.25),
        int(width * 0.56),
        int(height * 0.58),
    )
    gc_mask = np.zeros((height, width), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image, gc_mask, rect, bg_model, fg_model,
            5, cv2.GC_INIT_WITH_RECT
        )
        grabcut = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)
        save_result(image_path, image, "grabcut", grabcut)
    except cv2.error as exc:
        print("GrabCut failed:", image_path, exc)

    print(f"Processed {index}/{len(images)}: {image_path.name}")

csv_path = OUT / "experiment_outputs.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "method", "parameters",
                       "mask_path", "overlay_path"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nFinished.")
print("Images attempted:", len(images))
print("Prediction records:", len(results))
print("Results index:", csv_path)
print("Outputs:", OUT)
