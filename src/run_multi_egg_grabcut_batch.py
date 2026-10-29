from pathlib import Path
import json
import csv
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
META_ROOT = Path("outputs/sam_all/metadata")
MASK_ROOT = Path("outputs/sam_all/masks")
OUT = Path("outputs/multi_egg_grabcut_batch")

MASK_OUT = OUT / "masks"
OVERLAY_OUT = OUT / "overlays"
MASK_OUT.mkdir(parents=True, exist_ok=True)
OVERLAY_OUT.mkdir(parents=True, exist_ok=True)

# Use 10 images from the training set only.
train_images = sorted(
    p for p in (ROOT / "train" / "egg").rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)[:10]

if not train_images:
    raise SystemExit("No training images found.")

def locate_metadata(relative):
    # Match using the complete image path stored in metadata.
    for path in META_ROOT.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("image") == relative.as_posix():
            return path, data
    return None, None

def locate_sam_masks(relative):
    key = relative.with_suffix("")
    return (MASK_ROOT / key).with_suffix(".npz")

def refine_candidate(image, proposal):
    h, w = proposal.shape

    # Keep the largest connected component.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        proposal.astype(np.uint8), 8
    )
    if n <= 1:
        return None

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component = labels == largest

    area = int(component.sum())
    ratio = area / (h * w)

    # Exclude tiny regions and huge background-like regions.
    if ratio < 0.015 or ratio > 0.35:
        return None

    ys, xs = np.where(component > 0)

    if len(xs) == 0 or len(ys) == 0:
       return None

    x, y = int(xs.min()), int(ys.min())
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    aspect = bw / max(bh, 1)

    # Egg-like regions are generally elongated or oval.
    if not 0.45 <= aspect <= 2.2:
        return None

    # Initialize GrabCut using the SAM proposal.
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[component] = cv2.GC_PR_FGD

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    eroded = cv2.erode(
        component.astype(np.uint8), kernel, iterations=1
    ).astype(bool)
    dilated = cv2.dilate(
        component.astype(np.uint8), kernel, iterations=2
    ).astype(bool)

    gc_mask[dilated & ~component] = cv2.GC_PR_BGD
    gc_mask[eroded] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image, gc_mask, None, bg_model, fg_model,
            5, cv2.GC_INIT_WITH_MASK
        )
        refined = (
            (gc_mask == cv2.GC_FGD)
            | (gc_mask == cv2.GC_PR_FGD)
        )
    except cv2.error:
        refined = component

    # Constrain refinement to the local candidate neighborhood.
    refined &= dilated

    if refined.sum() < 0.5 * area:
        return component

    return refined

results = []

for i, image_path in enumerate(train_images, 1):
    relative = image_path.relative_to(ROOT)
    meta_path, metadata = locate_metadata(relative)

    if metadata is None:
        print(f"[{i}/10] Missing SAM metadata: {relative}")
        continue

    sam_path = locate_sam_masks(relative)
    if not sam_path.exists():
        print(f"[{i}/10] Missing SAM masks: {relative}")
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[{i}/10] Unreadable image: {relative}")
        continue

    with np.load(sam_path) as saved:
        proposals = saved["masks"]

    combined = np.zeros(image.shape[:2], dtype=bool)
    selected = 0

    # Filter every proposal instead of assuming candidates 0, 1, 2
    # are always eggs.
    for idx, proposal in enumerate(proposals):
        refined = refine_candidate(image, proposal)
        if refined is None:
            continue

        # Avoid adding candidates that mostly duplicate an existing mask.
        overlap = np.logical_and(combined, refined).sum()
        new_area = refined.sum()
        if new_area == 0:
            continue
        if overlap / new_area > 0.80:
            continue

        combined |= refined
        selected += 1

    mask_path = MASK_OUT / relative.parent / f"{relative.stem}.png"
    overlay_path = OVERLAY_OUT / relative.parent / f"{relative.stem}.jpg"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    binary = combined.astype(np.uint8) * 255
    cv2.imwrite(str(mask_path), binary)

    overlay = image.copy()
    overlay[combined] = (
        0.55 * image[combined] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    cv2.imwrite(str(overlay_path), overlay)

    results.append({
        "image": relative.as_posix(),
        "selected_candidates": selected,
        "foreground_pixels": int(combined.sum()),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    })

    print(
        f"[{i}/10] {relative.name}: "
        f"{selected} candidates, {int(combined.sum())} foreground pixels"
    )

csv_path = OUT / "batch_results.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image", "selected_candidates", "foreground_pixels",
            "mask_path", "overlay_path"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nCompleted:", len(results), "/", len(train_images))
print("Results:", csv_path)
print("Overlays:", OVERLAY_OUT)
print("Note: this is a heuristic experiment; inspect masks for errors.")
