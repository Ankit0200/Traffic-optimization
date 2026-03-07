"""
Aerial Vehicle Detection Test
===============================
Tests different YOLO models and settings to find what works best
for your aerial/drone intersection video.

Usage:
    python detect_aerial.py --video your_video.mp4

It will try multiple configurations and show you the results.
Press any key to move to next config, 'q' to quit.
"""

import cv2
import sys
from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--frame", type=int, default=50, help="Frame number to test on")
args = parser.parse_args()

# Grab a single frame to test on
cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Cannot read frame")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# ── Configs to test ───────────────────────────────────────────────────
configs = [
    {
        "name": "YOLOv8n pretrained — cars/bus/truck",
        "model": "yolov8n.pt",
        "classes": [2, 5, 7],
        "conf": 0.25
    },
    {
        "name": "YOLOv8m pretrained — cars/bus/truck",
        "model": "yolov8m.pt",
        "classes": [2, 5, 7],
        "conf": 0.25
    },
    {
        "name": "YOLOv8m pretrained — ALL classes",
        "model": "yolov8m.pt",
        "classes": None,
        "conf": 0.25
    },
    {
        "name": "YOLOv8m pretrained — low confidence",
        "model": "yolov8m.pt",
        "classes": [2, 5, 7],
        "conf": 0.10
    },
    {
        "name": "YOLOv8l pretrained — cars/bus/truck",
        "model": "yolov8l.pt",
        "classes": [2, 5, 7],
        "conf": 0.25
    },
    {
        "name": "Your custom model — class 0",
        "model": "10_epoch.pt",
        "classes": [0],
        "conf": 0.25
    },
    {
        "name": "Your custom model — low confidence",
        "model": "10_epoch.pt",
        "classes": [0],
        "conf": 0.10
    },
]

print(f"\nTesting {len(configs)} configurations...")
print("Press any key for next config, 'q' to quit\n")

for i, cfg in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] {cfg['name']}")
    print(f"  Model: {cfg['model']}, Classes: {cfg['classes']}, Conf: {cfg['conf']}")

    try:
        model = YOLO(cfg["model"])
    except Exception as e:
        print(f"  ⚠ Could not load model: {e}\n")
        continue

    # Run detection
    if cfg["classes"]:
        results = model(frame, conf=cfg["conf"], classes=cfg["classes"], verbose=False)
    else:
        results = model(frame, conf=cfg["conf"], verbose=False)

    # Draw results
    display = frame.copy()
    boxes = results[0].boxes
    num_detections = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        cls_name = model.names[cls]

        # Color by confidence
        green = int(conf * 255)
        color = (0, green, 255 - green)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{cls_name} {conf:.0%}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # HUD
    cv2.putText(display, f"[{i+1}/{len(configs)}] {cfg['name']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Detections: {num_detections}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Model: {cfg['model']} | Conf: {cfg['conf']} | Classes: {cfg['classes']}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, "Press any key for next, 'q' to quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    print(f"  → {num_detections} detections found\n")

    cv2.imshow("Detection Test", display)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Done! Use whichever config detected the most cars accurately.")
print("Then update your tracking pipeline with that model + settings.")