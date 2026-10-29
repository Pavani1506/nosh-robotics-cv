from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
META_ROOT = Path("outputs/sam_all/metadata")
MASK_ROOT = Path("outputs/sam_all/masks")
OUT_ROOT = Path("outputs/egg_candidates_v2")
PREVIEW_ROOT = Path("outputs/egg_candidate_previews_v2")

OUT_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)


def score_mask(mask, info, width, height):
    mask_u8 = mask.astype(np.uint8)
    area = int(mask_u8.sum())
    ratio = area / (width * height)

    if ratio < 0.015 or ratio > 0.65:
        return -1.0

    x, y, w, h = info["bbox"]
    if w <= 0 or h <= 0:
        return -1.0

    # Penalize regions touching image boundaries.
    edge_touch = (
        x <= 1 or y <= 1 or
        x + w >= width - 2 or y + h >= height - 2
    )
    edge_factor = 0.55 if edge_touch else 1.0

    # Prefer compact, approximately oval regions.
    aspect = w / h
    aspect_score = np.exp(-abs(np.log(aspect / 1.5)))

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return -1.0

    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    solidity = contour_area / max(hull_area, 1.0)

    # Check whether one connected component dominates the mask.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
    component_ratio = (
        component_areas.max() / max(area, 1)
        if len(component_areas) else 0.0
    )

    score = (
        0.35 * aspect_score
        + 0.35 * solidity
        + 0.30 * component_ratio
    ) * edge_factor

    return float(score)


selected = 0
no_candidate = 0

for meta_path in META_ROOT.rglob("*.json"):
    data = json.loads(meta_path.read_text())
    relative = Path(data["image"])
    key = relative.with_suffix("")

    mask_path = (MASK_ROOT / key).with_suffix(".npz")
    image_path = ROOT / relative

    if not mask_path.exists() or not image_path.exists():
        print("Missing input:", relative)
        no_candidate += 1
        continue

    with np.load(mask_path) as saved:
        masks = saved["masks"]

    image = cv2.imread(str(image_path))
    if image is None:
        no_candidate += 1
        continue

    height, width = image.shape[:2]
    candidates = []

    for info in data["masks"]:
        idx = info["index"]
        if idx >= len(masks):
            continue

        mask = masks[idx].astype(bool)
        score = score_mask(mask, info, width, height)
        if score >= 0:
            candidates.append((score, idx, mask))

    if not candidates:
        no_candidate += 1
        continue

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, idx, best_mask = candidates[0]

    output_dir = OUT_ROOT / relative.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_dir / f"{relative.stem}.png"),
        best_mask.astype(np.uint8) * 255
    )

    overlay = image.copy()
    overlay[best_mask] = (
        0.55 * overlay[best_mask]
        + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)

    preview_dir = PREVIEW_ROOT / relative.parent
    preview_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(preview_dir / f"{relative.stem}.jpg"),
        overlay
    )

    selected += 1

print("Images with selected candidates:", selected)
print("Images with no eligible candidate:", no_candidate)
print("Masks saved to:", OUT_ROOT)
print("Previews saved to:", PREVIEW_ROOT)
