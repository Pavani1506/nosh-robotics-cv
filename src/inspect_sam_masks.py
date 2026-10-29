from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path("data/egg.v1i.folder")
META_ROOT = Path("outputs/sam_all/metadata")
MASK_ROOT = Path("outputs/sam_all/masks")
OUT = Path("outputs/mask_inspection")
OUT.mkdir(parents=True, exist_ok=True)

metadata_files = sorted(META_ROOT.rglob("*.json"))
if not metadata_files:
    raise SystemExit("No SAM metadata files found.")

# Select one processed image for inspection.
meta_path = metadata_files[0]
data = json.loads(meta_path.read_text())

relative = Path(data["image"])
image_path = ROOT / relative
key = relative.with_suffix("")
mask_path = (MASK_ROOT / key).with_suffix(".npz")

image = cv2.imread(str(image_path))
if image is None or not mask_path.exists():
    raise SystemExit(f"Missing image or masks for {relative}")

with np.load(mask_path) as archive:
    masks = archive["masks"]

print("Image:", relative)
print("Number of masks:", len(masks))

# Save a contact sheet showing each candidate mask separately.
thumbs = []
for i, mask in enumerate(masks):
    binary = (mask > 0).astype(np.uint8) * 255
    thumb = cv2.resize(binary, (160, 160), interpolation=cv2.INTER_NEAREST)
    thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        thumb, f"Mask {i}", (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
    )
    thumbs.append(thumb)

columns = 4
rows = (len(thumbs) + columns - 1) // columns
sheet = np.zeros((rows * 160, columns * 160, 3), dtype=np.uint8)

for i, thumb in enumerate(thumbs):
    y = (i // columns) * 160
    x = (i % columns) * 160
    sheet[y:y + 160, x:x + 160] = thumb

cv2.imwrite(str(OUT / "candidate_masks.jpg"), sheet)
cv2.imwrite(str(OUT / "original.jpg"), image)

print("Saved:", OUT / "original.jpg")
print("Saved:", OUT / "candidate_masks.jpg")
