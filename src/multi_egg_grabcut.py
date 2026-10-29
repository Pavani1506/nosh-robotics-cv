from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
META_ROOT = Path("outputs/sam_all/metadata")
MASK_ROOT = Path("outputs/sam_all/masks")
OUT = Path("outputs/multi_egg_grabcut")
OUT.mkdir(parents=True, exist_ok=True)

# Start with the example containing three eggs.
TARGET = "not_damaged_103_"

meta_path = None
for path in META_ROOT.rglob("*.json"):
    data = json.loads(path.read_text())
    if TARGET in data.get("image", ""):
        meta_path = path
        break

if meta_path is None:
    raise SystemExit("Could not find the target image metadata.")

data = json.loads(meta_path.read_text())
relative = Path(data["image"])
image_path = ROOT / relative
mask_path = (MASK_ROOT / relative.with_suffix("")).with_suffix(".npz")

image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit(f"Cannot read image: {image_path}")

with np.load(mask_path) as saved:
    proposals = saved["masks"]

height, width = image.shape[:2]
combined = np.zeros((height, width), dtype=np.uint8)
debug = []

# These are the three plausible candidates identified during inspection.
for idx in (0, 1, 2):
    if idx >= len(proposals):
        continue

    proposal = proposals[idx].astype(bool)
    if not proposal.any():
        continue

    # Keep only the largest connected component of this proposal.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        proposal.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        continue

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component = labels == largest_label

    # Initialize GrabCut: SAM foreground inside the proposal,
    # probable background in a narrow region around it.
    gc_mask = np.full(
        (height, width), cv2.GC_BGD, dtype=np.uint8
    )
    gc_mask[component] = cv2.GC_PR_FGD

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    eroded = cv2.erode(
        component.astype(np.uint8), kernel, iterations=1
    ).astype(bool)

    # Use the proposal's eroded interior as definite foreground.
    if eroded.any():
        gc_mask[eroded] = cv2.GC_FGD

    # Treat a narrow outer neighborhood as probable background.
    dilated = cv2.dilate(
        component.astype(np.uint8), kernel, iterations=2
    ).astype(bool)
    gc_mask[dilated & ~component] = cv2.GC_PR_BGD

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
    except cv2.error as exc:
        print(f"GrabCut failed for candidate {idx}: {exc}")
        refined = component

    combined[refined] = 255
    debug.append((idx, component, refined))

# Save the combined binary mask.
mask_out = OUT / "not_damaged_103_combined.png"
cv2.imwrite(str(mask_out), combined)

# Save an overlay of all selected/refined regions.
overlay = image.copy()
selected = combined > 0
overlay[selected] = (
    0.55 * image[selected]
    + 0.45 * np.array([0, 255, 0])
).astype(np.uint8)

overlay_out = OUT / "not_damaged_103_overlay.jpg"
cv2.imwrite(str(overlay_out), overlay)

# Save a diagnostic showing each candidate before and after refinement.
panels = []
for idx, before, after in debug:
    before_img = np.zeros_like(image)
    before_img[before] = (255, 255, 255)
    after_img = np.zeros_like(image)
    after_img[after] = (255, 255, 255)

    before_img = cv2.resize(before_img, (320, 320))
    after_img = cv2.resize(after_img, (320, 320))
    panels.extend([before_img, after_img])

if panels:
    diagnostic = np.hstack(panels)
    cv2.imwrite(str(OUT / "candidate_diagnostic.png"), diagnostic)

print("Input image:", relative)
print("Candidate masks processed:", len(debug))
print("Combined mask:", mask_out)
print("Overlay:", overlay_out)
print("Diagnostic:", OUT / "candidate_diagnostic.png")
print("Review the results visually before using them for evaluation.")
