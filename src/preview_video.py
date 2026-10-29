import cv2
from pathlib import Path

video_path = "data/video/egg_video.mp4"
output_dir = Path("outputs/video_tracking/preview")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {video_path}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Frames: {total}, FPS: {fps}")

for i, fraction in enumerate([0.05, 0.20, 0.40, 0.60, 0.80, 0.95]):
    frame_no = int(total * fraction)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()

    if ok:
        path = output_dir / f"frame_{frame_no:04d}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")

cap.release()
