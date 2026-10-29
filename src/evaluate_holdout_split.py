import csv
from pathlib import Path

import cv2
import numpy as np

MANIFEST = Path("reports/dev_eval_split.csv")
REFERENCE_ROOT = Path("outputs/automated_evaluation")
PREDICTION_ROOT = Path("outputs/multi_egg_grabcut_batch_v2/masks")
OUT = Path("outputs/automated_evaluation/holdout_metrics.csv")


def index_masks(root):
    result = {}
    for path in root.rglob("*.png"):
        key = path.stem.replace("_selected", "")
        result[key] = path
    return result


references = index_masks(REFERENCE_ROOT)
predictions = index_masks(PREDICTION_ROOT)
rows = []

with MANIFEST.open(newline="") as f:
    manifest = csv.DictReader(f)

    for item in manifest:
        if item["split"] != "evaluation":
            continue

        image_key = Path(item["image"]).stem
        reference_path = references.get(image_key)
        prediction_path = predictions.get(image_key)

        if reference_path is None or prediction_path is None:
            rows.append({
                "image": image_key,
                "iou": "",
                "dice": "",
                "status": "missing reference or prediction",
            })
            continue

        ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)

        if ref is None or pred is None:
            continue

        if ref.shape != pred.shape:
            pred = cv2.resize(
                pred, (ref.shape[1], ref.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        ref = ref > 0
        pred = pred > 0

        intersection = np.logical_and(ref, pred).sum()
        union = np.logical_or(ref, pred).sum()
        denominator = ref.sum() + pred.sum()

        rows.append({
            "image": image_key,
            "iou": round(float(intersection / union), 4) if union else 1.0,
            "dice": round(float(2 * intersection / denominator), 4)
                    if denominator else 1.0,
            "status": "compared with selected SAM pseudo-reference",
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "iou", "dice", "status"]
    )
    writer.writeheader()
    writer.writerows(rows)

valid = [r for r in rows if isinstance(r["iou"], float)]

print("Evaluation images in manifest:", sum(
    1 for r in rows
))
print("Images compared:", len(valid))
print("Missing comparisons:", len(rows) - len(valid))
if valid:
    print("Mean IoU:", round(np.mean([r["iou"] for r in valid]), 4))
    print("Mean Dice:", round(np.mean([r["dice"] for r in valid]), 4))
print("Saved:", OUT)
print("Reminder: scores measure agreement with selected SAM masks, not ground truth.")
