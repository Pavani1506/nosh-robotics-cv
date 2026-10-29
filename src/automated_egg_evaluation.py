import csv
import json
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("data/egg.v1i.folder")
SAM_ROOT = Path("outputs/sam_all/masks")
PRED_ROOT = Path("outputs/multi_egg_grabcut_batch_v2/masks")
OUT_ROOT = Path("outputs/automated_evaluation")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.35
MIN_ASPECT = 0.45
MAX_ASPECT = 2.2
MIN_SOLIDITY = 0.65
OVERLAP_THRESHOLD = 0.80


def load_sam_candidates(image_path):
    relative = image_path.relative_to(DATA_ROOT)
    archive = (SAM_ROOT / relative.with_suffix("")).with_suffix(".npz")

    if not archive.exists():
        return None

    with np.load(archive) as data:
        masks = data["masks"]

    return masks > 0


def select_egg_candidates(candidates):
    height, width = candidates.shape[1:]
    image_area = height * width
    accepted = []

    for index, mask in enumerate(candidates):
        count = int(mask.sum())
        ratio = count / image_area

        if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
            continue

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        box_width, box_height = x2 - x1, y2 - y1

        if min(box_width, box_height) == 0:
            continue

        aspect = box_width / box_height
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            continue

        if x1 == 0 or y1 == 0 or x2 >= width or y2 >= height:
            continue

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))

        solidity = contour_area / hull_area if hull_area else 0
        if solidity < MIN_SOLIDITY:
            continue

        accepted.append((index, mask, count))

    # Keep the larger mask when candidates overlap heavily.
    accepted.sort(key=lambda item: item[2], reverse=True)
    selected = []

    for candidate in accepted:
        _, mask, _ = candidate
        duplicate = False

        for _, chosen_mask, _ in selected:
            intersection = np.logical_and(mask, chosen_mask).sum()
            union = np.logical_or(mask, chosen_mask).sum()
            iou = intersection / union if union else 0

            if iou > OVERLAP_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

    return selected


def main():
    records = []

    for image_path in sorted((DATA_ROOT / "train" / "egg").glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        candidates = load_sam_candidates(image_path)
        if candidates is None:
            print("Missing SAM archive:", image_path.name)
            continue

        selected = select_egg_candidates(candidates)
        combined = np.zeros(candidates.shape[1:], dtype=np.uint8)

        for _, mask, _ in selected:
            combined[mask] = 1

        output_path = OUT_ROOT / f"{image_path.stem}_selected.png"
        cv2.imwrite(str(output_path), combined * 255)

        records.append({
            "image": str(image_path.relative_to(DATA_ROOT)),
            "sam_candidate_count": len(candidates),
            "selected_candidate_count": len(selected),
            "selected_pixels": int(combined.sum()),
            "output_mask": str(output_path),
        })

        print(
            f"{image_path.name}: "
            f"{len(selected)} candidate(s) selected"
        )

    csv_path = OUT_ROOT / "selection_summary.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "sam_candidate_count",
                "selected_candidate_count",
                "selected_pixels",
                "output_mask",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "images_processed": len(records),
        "parameters": {
            "min_area_ratio": MIN_AREA_RATIO,
            "max_area_ratio": MAX_AREA_RATIO,
            "min_aspect": MIN_ASPECT,
            "max_aspect": MAX_ASPECT,
            "min_solidity": MIN_SOLIDITY,
            "overlap_threshold": OVERLAP_THRESHOLD,
        },
        "warning": (
            "Automated SAM proposal filtering is a heuristic. "
            "Selected masks are not verified ground truth."
        ),
    }

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nFinished.")
    print("Summary:", OUT_ROOT / "summary.json")
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
