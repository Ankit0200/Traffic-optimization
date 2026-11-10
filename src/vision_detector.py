"""
vision_detector.py — YOLO-based Vehicle Detection for SUMO Screenshots
======================================================================
Captures screenshots from SUMO GUI via TraCI, runs YOLOv8 detection
with built-in tracking, and returns vehicle positions in pixel/grid
coordinates.

This replaces TraCI ground-truth positions with noisy vision detections
so the adaptive controller is genuinely vision-based.

Usage:
    detector = VisionDetector(model_path="models/yolov8n-visdrone.pt")
    detections = detector.detect(conn, width=1920, height=1080)
    # detections = [{"track_id": 1, "center": (px, py), "bbox": ...}, ...]
"""

import os
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO


# VisDrone class IDs for vehicles
VISDRONE_VEHICLE_CLASSES = [3, 4, 5, 8, 9]  # car, van, truck, bus, motor

# COCO class IDs for vehicles
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


class VisionDetector:
    """
    Wraps YOLOv8 detection + tracking on SUMO GUI screenshots.

    The detector:
    1. Captures a screenshot from SUMO GUI via traci.gui.screenshot()
    2. Runs YOLOv8 with built-in BoT-SORT tracking (persist=True)
    3. Returns tracked vehicle detections with pixel coordinates

    Track IDs persist across frames (handled by YOLO's tracker).
    """

    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5,
                 vehicle_classes=None):
        """
        Args:
            model_path:      Path to YOLOv8 .pt model file
            conf_threshold:  Minimum detection confidence
            iou_threshold:   NMS IoU threshold
            vehicle_classes: List of class IDs to detect. Auto-detected from
                             model name if None (visdrone vs COCO).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold

        # Auto-detect vehicle classes from model name
        if vehicle_classes is not None:
            self.classes = vehicle_classes
        elif "visdrone" in os.path.basename(model_path).lower():
            self.classes = VISDRONE_VEHICLE_CLASSES
        else:
            self.classes = COCO_VEHICLE_CLASSES

        # Temp file for screenshots (reused to avoid filesystem churn)
        self._tmp_dir = tempfile.mkdtemp(prefix="sumo_vision_")
        self._tmp_path = os.path.join(self._tmp_dir, "frame.png")

        # Track state
        self._frame_count = 0

    def detect(self, conn, view_id="View #0", width=1920, height=1080):
        """
        Capture screenshot from SUMO GUI and run YOLO detection.

        Args:
            conn:    TraCI connection (must be sumo-gui, not headless sumo)
            view_id: GUI view identifier
            width:   Screenshot width in pixels
            height:  Screenshot height in pixels

        Returns:
            List of detection dicts:
            [{"track_id": int, "center": (px, py),
              "bbox": (x1, y1, x2, y2), "conf": float}, ...]
        """
        # Capture screenshot
        conn.gui.screenshot(view_id, self._tmp_path, width, height)

        # Read the image
        frame = cv2.imread(self._tmp_path)
        if frame is None:
            return []

        self._frame_count += 1

        # Run YOLO tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )

        detections = []
        if (results and results[0].boxes is not None
                and results[0].boxes.id is not None):
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                conf = float(boxes.conf[i])
                detections.append({
                    "track_id": track_id,
                    "center": (cx, cy),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "conf": conf,
                })

        return detections

    def cleanup(self):
        """Remove temp files."""
        try:
            if os.path.exists(self._tmp_path):
                os.remove(self._tmp_path)
            if os.path.exists(self._tmp_dir):
                os.rmdir(self._tmp_dir)
        except OSError:
            pass

    def __del__(self):
        self.cleanup()
