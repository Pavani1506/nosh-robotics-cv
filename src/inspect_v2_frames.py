import cv2
from pathlib import Path

video = "outputs/video_tracking/v2/egg_tracking_v2.mp4"
out = Path("outputs/video_tracking/v2/inspection")
out.mkdir(parents=True, exist_ok=True)

frames = [647, 649, 652, 732, 734, 737, 809, 818, 821]
cap = cv2.VideoCapture(video)

for n in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    if ok:
        path = out / f"frame_{n}.jpg"
        cv2.imwrite(str(path), frame)
        print(path)

cap.release()
