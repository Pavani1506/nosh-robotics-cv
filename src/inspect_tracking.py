import cv2
from pathlib import Path

video = "outputs/video_tracking/egg_tracking.mp4"
out = Path("outputs/video_tracking/inspection")
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError("Cannot open annotated video")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
positions = [0.05, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
tiles = []

for fraction in positions:
    frame_no = int(total * fraction)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.resize(frame, (480, 270))
    cv2.putText(frame, f"Frame {frame_no}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(frame)

cap.release()

if not tiles:
    raise RuntimeError("No frames extracted")

sheet = cv2.vconcat([
    cv2.hconcat(tiles[i:i+2])
    for i in range(0, len(tiles), 2)
])
path = out / "tracking_contact_sheet.jpg"
cv2.imwrite(str(path), sheet)
print(f"Saved: {path}")
