
from pathlib import Path
import csv
import json
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
META_ROOT = Path("outputs/sam_all/metadata")
SAM_MASK_ROOT = Path("outputs/sam_all/masks")
RESULTS_CSV = Path(
    "outputs/multi_egg_grabcut_batch_v2/batch_results.csv"
)
OUT_CSV = Path(
    "outputs/multi_egg_grabcut_batch_v2/evaluation_metrics.csv"
)


def find_sam_reference(relative):
    for meta_path in META_ROOT.rglob("*.json"):
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        if data.get("image") == relative.as_posix():
            # Match the filename convention used by the existing SAM run.
            key = relative.with_suffix("")
            mask_path = (SAM_MASK_ROOT / key).with_suffix(".npz")
            if mask_path.exists():
                return mask_path
            return None

    return None


def load_reference_union(npz_path, image_shape):
    with np.load(npz_path, allow_pickle=True) as data:
        # Inspect available array names without assuming the archive schema.
        keys = list(data.files)
        if not keys:
            raise ValueError(f"No arrays in {npz_path}")

        # Common SAM automatic-mask export format.
        if "masks" in data:
            proposals = data["masks"]
        elif "arr_0" in data:
            proposals = data["arr_0"]
        else:
            raise ValueError(
                f"Unknown SAM archive schema in {npz_path}: {keys}"
            )

        proposals = np.asarray(proposals)

        # Expected format: N x H x W boolean/binary masks.
        if proposals.ndim == 2:
            union = proposals > 0
        elif proposals.ndim == 3:
            union = np.any(proposals > 0, axis=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {proposals.shape} in {npz_path}"
            )

    if union.shape != image_shape:
        union = cv2.resize(
            union.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    return union


def scores(pred, ref):
    pred = pred.astype(bool)
    ref = ref.astype(bool)

    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    total = pred.sum() + ref.sum()

    iou = intersection / union if union else 1.0
    dice = (2 * intersection) / total if total else 1.0
    return float(iou), float(dice)


rows = []

with RESULTS_CSV.open(newline="") as f:
    for item in csv.DictReader(f):
        relative = Path(item["image"])
        pred_path = Path(item["mask_path"])
        image_path = ROOT / relative

        image = cv2.imread(str(image_path))
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

        if image is None or pred is None:
            print(f"SKIP: Could not load image or prediction: {relative}")
            continue

        sam_path = find_sam_reference(relative)
        if sam_path is None:
            print(f"SKIP: SAM reference not found: {relative}")
            continue

        try:
            reference = load_reference_union(sam_path, image.shape[:2])
        except (ValueError, OSError) as exc:
            print(f"SKIP: {relative}: {exc}")
            continue

        if pred.shape != reference.shape:
            pred = cv2.resize(
                pred,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        iou, dice = scores(pred > 0, reference)

        rows.append({
            "image": relative.as_posix(),
            "iou_vs_sam": round(iou, 5),
            "dice_vs_sam": round(dice, 5),
            "predicted_pixels": int((pred > 0).sum()),
            "sam_reference_pixels": int(reference.sum()),
        })

with OUT_CSV.open("w", newline="") as f:
    fields = [
        "image",
        "iou_vs_sam",
        "dice_vs_sam",
        "predicted_pixels",
        "sam_reference_pixels",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Evaluated images: {len(rows)}")
if rows:
    print(f"Mean IoU vs SAM: {np.mean([r['iou_vs_sam'] for r in rows]):.4f}")
    print(f"Mean Dice vs SAM: {np.mean([r['dice_vs_sam'] for r in rows]):.4f}")
print(f"Saved metrics: {OUT_CSV}")

