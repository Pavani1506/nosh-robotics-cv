import cv2
import csv
import numpy as np
from pathlib import Path
from math import hypot

VIDEO = "data/video/egg_video.mp4"
OUT = Path("outputs/video_tracking")
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 24
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUT / "egg_tracking.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# Track state: ID -> {"center": (x, y), "missed": count}
tracks = {}
next_id = 1
MAX_DISTANCE = 65
MAX_MISSED = 8

csv_file = open(OUT / "tracks.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "time_seconds", "track_id", "center_x", "center_y",
                     "bbox_x", "bbox_y", "bbox_width", "bbox_height", "area"])

frame_no = 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    display = frame.copy()

    # Threshold bright, low-saturation regions in HSV.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 115, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find egg-sized, approximately oval contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2500 or area > 65000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            continue

        aspect = w / h
        if not 0.55 <= aspect <= 1.8:
            continue

        # Reject contours that are too thin or irregular.
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.35:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        detections.append({
            "center": (cx, cy), "bbox": (x, y, w, h), "area": int(area)
        })

    # Increase missed count before matching this frame.
    for track in tracks.values():
        track["missed"] += 1

    # Greedy nearest-centroid matching.
    unmatched = set(range(len(detections)))
    for track_id, track in list(tracks.items()):
        tx, ty = track["center"]
        candidates = [
            (hypot(tx - d["center"][0], ty - d["center"][1]), i)
            for i, d in enumerate(detections) if i in unmatched
        ]
        if candidates:
            distance, idx = min(candidates)
            if distance <= MAX_DISTANCE:
                track["center"] = detections[idx]["center"]
                track["missed"] = 0
                detections[idx]["track_id"] = track_id
                unmatched.remove(idx)

    for idx in sorted(unmatched):
        detections[idx]["track_id"] = next_id
        tracks[next_id] = {"center": detections[idx]["center"], "missed": 0}
        next_id += 1

    tracks = {
        tid: t for tid, t in tracks.items()
        if t["missed"] <= MAX_MISSED
    }

    for d in detections:
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        tid = d["track_id"]

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(display, f"Egg {tid}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

        csv_writer.writerow([
            frame_no, round((frame_no - 1) / fps, 3), tid, cx, cy,
            x, y, w, h, d["area"]
        ])

    cv2.putText(display, f"Frame {frame_no} | Detections: {len(detections)}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

    writer.write(display)

    if frame_no % 200 == 0:
        print(f"Processed {frame_no} frames")

cap.release()
writer.release()
csv_file.close()

print(f"Finished. Video: {OUT / 'egg_tracking.mp4'}")
print(f"Tracking CSV: {OUT / 'tracks.csv'}")
