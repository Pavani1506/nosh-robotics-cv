import cv2
import csv
import numpy as np
from pathlib import Path
from math import hypot

VIDEO = "data/video/egg_video.mp4"
OUT = Path("outputs/video_tracking/v2")
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 24
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUT / "egg_tracking_v2.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
)

# Restrict detection to the tabletop.
# This is a starting estimate; verify against the original video.
TABLE_Y = int(H * 0.39)

tracks = {}
next_id = 1
MAX_DISTANCE = 90
MAX_MISSED = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

csv_file = open(OUT / "tracks_v2.csv", "w", newline="")
log = csv.writer(csv_file)
log.writerow([
    "frame", "time_seconds", "track_id", "center_x", "center_y",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height", "area"
])

frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    display = frame.copy()

    # Analyze only the tabletop.
    roi = frame[TABLE_Y:H, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # White eggs: high brightness and low saturation.
    mask = cv2.inRange(
        hsv, np.array([0, 0, 155]), np.array([179, 105, 255])
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 6000 <= area <= 95000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) == 0:
            continue

        aspect = w / h
        if not 0.65 <= aspect <= 1.65:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.48:
            continue

        # Offset ROI coordinates back to full-frame coordinates.
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + TABLE_Y
        detections.append({
            "center": (cx, cy),
            "bbox": (x, y + TABLE_Y, w, h),
            "area": int(area)
        })

    # Match existing tracks to the nearest unused detection.
    for t in tracks.values():
        t["missed"] += 1

    unmatched = set(range(len(detections)))
    pairs = []

    for tid, t in tracks.items():
        tx, ty = t["center"]
        for i in unmatched:
            dx, dy = detections[i]["center"]
            dist = hypot(tx - dx, ty - dy)
            if dist <= MAX_DISTANCE:
                pairs.append((dist, tid, i))

    used_tracks = set()
    for _, tid, i in sorted(pairs):
        if tid in used_tracks or i not in unmatched:
            continue
        tracks[tid]["center"] = detections[i]["center"]
        tracks[tid]["missed"] = 0
        detections[i]["track_id"] = tid
        unmatched.remove(i)
        used_tracks.add(tid)

    # New IDs only for detections inside the tabletop ROI.
    for i in sorted(unmatched):
        detections[i]["track_id"] = next_id
        tracks[next_id] = {
            "center": detections[i]["center"], "missed": 0
        }
        next_id += 1

    tracks = {
        tid: t for tid, t in tracks.items()
        if t["missed"] <= MAX_MISSED
    }

    for d in detections:
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        tid = d["track_id"]

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 3)
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            display, f"Egg {tid}", (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2
        )
        log.writerow([
            frame_no, round((frame_no - 1) / fps, 3), tid, cx, cy,
            x, y, w, h, d["area"]
        ])

    cv2.line(display, (0, TABLE_Y), (W, TABLE_Y), (255, 180, 0), 2)
    cv2.putText(
        display, f"Frame {frame_no} | Detections: {len(detections)}",
        (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 255), 2
    )
    writer.write(display)

    if frame_no % 200 == 0:
        print(f"Processed {frame_no}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

cap.release()
writer.release()
csv_file.close()

print("Video:", OUT / "egg_tracking_v2.mp4")
print("CSV:", OUT / "tracks_v2.csv")
