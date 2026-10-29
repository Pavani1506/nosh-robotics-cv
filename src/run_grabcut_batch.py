from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
OUT = Path("outputs/classical_batch")
MASK_DIR = OUT / "masks" / "grabcut"
OVERLAY_DIR = OUT / "overlays" / "grabcut"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted(
    p for p in (ROOT / "train" / "egg").rglob("*")
    if p.suffix.lower() in extensions
)[:10]

if not images:
    raise SystemExit("No training images found.")

results = []

for i, image_path in enumerate(images, start=1):
    image = cv2.imread(str(image_path))
    if image is None:
        print("Skipping unreadable image:", image_path)
        continue

    h, w = image.shape[:2]

    # Initial rectangle: same parameters as the existing GrabCut trial.
    rect = (
        int(w * 0.27), int(h * 0.25),
        int(w * 0.56), int(h * 0.58)
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image, mask, rect, bg_model, fg_model,
            5, cv2.GC_INIT_WITH_RECT
        )
    except cv2.error as exc:
        print("GrabCut failed:", image_path.name, exc)
        continue

    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    relative = image_path.relative_to(ROOT)
    mask_path = MASK_DIR / relative.parent / f"{relative.stem}.png"
    overlay_path = OVERLAY_DIR / relative.parent / f"{relative.stem}.jpg"

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(mask_path), binary)

    overlay = image.copy()
    selected = binary > 0
    overlay[selected] = (
        0.55 * image[selected] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    cv2.imwrite(str(overlay_path), overlay)

    results.append({
        "image": str(relative),
        "method": "GrabCut",
        "rectangle": str(rect),
        "iterations": 5,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    })

    print(f"Processed {i}/{len(images)}: {image_path.name}")

csv_path = OUT / "grabcut_batch_results.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image", "method", "rectangle", "iterations",
        "mask_path", "overlay_path"
    ])
    writer.writeheader()
    writer.writerows(results)

print("\nSuccessful images:", len(results), "/", len(images))
print("Results index:", csv_path)
print("Overlays:", OVERLAY_DIR)
