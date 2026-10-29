from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
META_ROOT = Path("outputs/sam_all/metadata")
MASK_ROOT = Path("outputs/sam_all/masks")
OUT_ROOT = Path("outputs/egg_candidates")
PREVIEW_ROOT = Path("outputs/egg_candidate_previews")

OUT_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)


def mask_quality(mask, info, width, height):
    """Heuristic ranking only; scores are not model probabilities."""
    area = int(mask.sum())
    image_area = width * height
    area_ratio = area / image_area

    x, y, w, h = info["bbox"]
    touches_edge = (
        x <= 1 or y <= 1 or
        x + w >= width - 2 or y + h >= height - 2
    )

    # Reject extremely small or image-dominating regions.
    if area_ratio < 0.015 or area_ratio > 0.65:
        return -1.0

    # Egg candidates are often compact, oval-like regions.
    aspect = w / max(h, 1)
    aspect_score = np.exp(-abs(np.log(max(aspect, 1e-6) / 1.5)))
    area_score = np.exp(-abs(np.log(max(area_ratio, 1e-6) / 0.15)))
    edge_penalty = 0.5 if touches_edge else 1.0

    return float(
        edge_penalty
        * (0.55 * aspect_score + 0.45 * area_score)
    )


selected = 0
no_candidate = 0

for meta_path in META_ROOT.rglob("*.json"):
    data = json.loads(meta_path.read_text())
    relative = Path(data["image"])
    key = relative.with_suffix("")

    # Matches the filename convention used by the completed SAM run.
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
        print("Cannot read image:", image_path)
        no_candidate += 1
        continue

    height, width = image.shape[:2]
    candidates = []

    for info in data["masks"]:
        idx = info["index"]
        if idx >= len(masks):
            continue

        mask = masks[idx].astype(bool)
        score = mask_quality(mask, info, width, height)
        if score >= 0:
            candidates.append((score, idx, mask))

    if not candidates:
        no_candidate += 1
        continue

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, idx, best_mask = candidates[0]

    # Preserve the full image stem to avoid truncating dotted filenames.
    output_dir = OUT_ROOT / relative.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{relative.stem}.png"

    cv2.imwrite(str(output_path), best_mask.astype(np.uint8) * 255)

    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :] = (0, 255, 0)
    overlay[best_mask] = (
        0.55 * overlay[best_mask] + 0.45 * color[best_mask]
    ).astype(np.uint8)

    preview_dir = PREVIEW_ROOT / relative.parent
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{relative.stem}.jpg"
    cv2.imwrite(str(preview_path), overlay)

    selected += 1

print("Images with selected candidate masks:", selected)
print("Images with no eligible candidate:", no_candidate)
print("Candidate masks:", OUT_ROOT)
print("Overlay previews:", PREVIEW_ROOT)
print("Important: candidates are heuristic pseudo-labels; inspect before evaluation.")
