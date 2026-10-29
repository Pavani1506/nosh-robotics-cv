from pathlib import Path
import json
import traceback

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

ROOT = Path("data/egg.v1i.folder")
CHECKPOINT = Path("models/sam_vit_b_01ec64.pth")
OUTPUT = Path("outputs/sam_all")

MASK_DIR = OUTPUT / "masks"
META_DIR = OUTPUT / "metadata"
OVERLAY_DIR = OUTPUT / "overlays"

for folder in (MASK_DIR, META_DIR, OVERLAY_DIR):
    folder.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if not images:
    raise SystemExit(f"No images found under {ROOT.resolve()}")

if not CHECKPOINT.exists():
    raise SystemExit(f"SAM checkpoint not found: {CHECKPOINT.resolve()}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
print(f"Images found: {len(images)}")

print("Loading SAM ViT-B model...")
sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
sam.to(device=device)

generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.90,
    crop_n_layers=0,
    min_mask_region_area=100,
)

success = 0
failures = []

for index, image_path in enumerate(images, start=1):
    relative = image_path.relative_to(ROOT)
    key = relative.with_suffix("")
    mask_path = (MASK_DIR / key).with_suffix(".npz")
    meta_path = (META_DIR / key).with_suffix(".json")
    overlay_path = (OVERLAY_DIR / key).with_suffix(".jpg")

    if mask_path.exists() and meta_path.exists():
        print(f"[{index}/{len(images)}] Already done: {relative}")
        success += 1
        continue

    try:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError("OpenCV could not read this image")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image_rgb)

        # Keep the segmentation masks as binary arrays in a compressed file.
        if masks:
            mask_arrays = np.stack([
                item["segmentation"].astype(np.uint8)
                for item in masks
            ])
        else:
            mask_arrays = np.zeros(
                (0, image_rgb.shape[0], image_rgb.shape[1]),
                dtype=np.uint8,
            )

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(mask_path, masks=mask_arrays)

        metadata = {
            "image": str(relative),
            "width": int(image_rgb.shape[1]),
            "height": int(image_rgb.shape[0]),
            "mask_count": len(masks),
            "masks": [
                {
                    "index": i,
                    "area": int(item["area"]),
                    "bbox": [float(v) for v in item["bbox"]],
                    "predicted_iou": float(item["predicted_iou"]),
                    "stability_score": float(item["stability_score"]),
                }
                for i, item in enumerate(masks)
            ],
        }
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Save an overlay for visual quality checks.
        overlay = image_rgb.copy()
        rng = np.random.default_rng(42)
        colors = rng.integers(
            50, 256, size=(len(masks), 3), dtype=np.uint8
        )
        for i, item in enumerate(masks):
            region = item["segmentation"]
            overlay[region] = (
                0.55 * overlay[region] + 0.45 * colors[i]
            ).astype(np.uint8)

        cv2.imwrite(
            str(overlay_path),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        success += 1
        print(
            f"[{index}/{len(images)}] OK: {relative} "
            f"({len(masks)} masks)"
        )

    except Exception as exc:
        failures.append({
            "image": str(relative),
            "error": repr(exc),
        })
        print(f"[{index}/{len(images)}] FAILED: {relative}: {exc}")
        traceback.print_exc()

(OUTPUT / "run_summary.json").write_text(json.dumps({
    "total_images": len(images),
    "successful_images": success,
    "failed_images": failures,
    "device": device,
    "parameters": {
        "points_per_side": 8,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.90,
        "crop_n_layers": 0,
        "min_mask_region_area": 100,
    },
}, indent=2))

print("\nFinished.")
print(f"Successful images: {success}/{len(images)}")
print(f"Failures: {len(failures)}")
print(f"Results saved under: {OUTPUT.resolve()}")
