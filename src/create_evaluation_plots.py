import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/automated_evaluation")
DATA = Path("data/egg.v1i.folder")
PRED = Path("outputs/multi_egg_grabcut_batch_v2/masks")
OUT = ROOT / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "grabcut_vs_selected_sam.csv", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("The evaluation CSV has no rows.")

ious = [float(r["iou_vs_selected_sam"]) for r in rows]
dices = [float(r["dice_vs_selected_sam"]) for r in rows]

# Plot 1: per-image metrics
x = np.arange(len(rows))
plt.figure(figsize=(12, 5))
plt.bar(x - 0.2, ious, width=0.4, label="IoU")
plt.bar(x + 0.2, dices, width=0.4, label="Dice")
plt.xticks(x, [r["image"][:17] for r in rows], rotation=60, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("GrabCut vs. selected SAM masks (10 matched images)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "per_image_metrics.png", dpi=160)
plt.close()

# Plot 2: visual comparison for the three lowest IoU cases
pred_map = {
    p.stem.replace("_selected", ""): p
    for p in PRED.rglob("*.png")
}

panels = []
for row in sorted(rows, key=lambda r: float(r["iou_vs_selected_sam"]))[:3]:
    key = row["image"]
    original_path = DATA / "train" / "egg" / f"{key}.jpg"
    if not original_path.exists():
        original_path = DATA / "train" / "egg" / f"{key}.png"

    pred_path = pred_map.get(key)
    ref_path = ROOT / f"{key}_selected.png"

    original = cv2.imread(str(original_path))
    reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) if pred_path else None

    if original is None or reference is None or prediction is None:
        continue

    if prediction.shape != reference.shape:
        prediction = cv2.resize(
            prediction, (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    ref = reference > 0
    pred = prediction > 0

    # Green = overlap, red = prediction-only, blue = reference-only
    comparison = np.zeros_like(original)
    comparison[np.logical_and(ref, pred)] = (0, 200, 0)
    comparison[np.logical_and(pred, ~ref)] = (0, 0, 255)
    comparison[np.logical_and(ref, ~pred)] = (255, 0, 0)

    original = cv2.resize(original, (320, 320))
    comparison = cv2.resize(comparison, (320, 320), interpolation=cv2.INTER_NEAREST)
    ref_vis = cv2.resize((ref * 255).astype(np.uint8), (320, 320))
    ref_vis = cv2.cvtColor(ref_vis, cv2.COLOR_GRAY2BGR)

    for panel, label in [
        (original, "Original"),
        (ref_vis, "Selected SAM"),
        (comparison, f'Comparison IoU={row["iou_vs_selected_sam"]}'),
    ]:
        panel = panel.copy()
        cv2.putText(panel, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1)
        panels.append(panel)

if panels:
    while len(panels) % 3:
        panels.append(np.zeros_like(panels[0]))
    sheet = cv2.vconcat([
        cv2.hconcat(panels[i:i+3])
        for i in range(0, len(panels), 3)
    ])
    cv2.imwrite(str(OUT / "failure_case_comparison.jpg"), sheet)

print("Saved report assets to:", OUT)
print("Mean IoU:", round(float(np.mean(ious)), 4))
print("Mean Dice:", round(float(np.mean(dices)), 4))
