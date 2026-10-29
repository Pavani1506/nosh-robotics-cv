from pathlib import Path
import cv2
import numpy as np

image_path = Path(
    "data/egg.v1i.folder/train/egg/"
    "not_damaged_108_jpg.rf.de0bd2447ade8d891625ac0dd055667a.jpg"
)
mask_path = Path(
    "outputs/sam_all/masks/train/egg/not_damaged_108_jpg.rf.npz"
)
out_dir = Path("outputs/egg_candidate_inspection")
out_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit(f"Image not found: {image_path}")

with np.load(mask_path) as data:
    masks = data["masks"]

tiles = []
for i, mask in enumerate(masks):
    overlay = image.copy()
    selected = mask > 0
    overlay[selected] = (
        0.5 * overlay[selected] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    cv2.putText(
        overlay, f"Candidate {i}", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    tiles.append(cv2.resize(overlay, (320, 320)))

canvas = np.zeros((960, 1280, 3), dtype=np.uint8)
for i, tile in enumerate(tiles):
    row, col = divmod(i, 4)
    canvas[row * 320:(row + 1) * 320,
           col * 320:(col + 1) * 320] = tile

out_path = out_dir / "not_damaged_108_candidates.jpg"
cv2.imwrite(str(out_path), canvas)
print(f"Saved: {out_path}")
