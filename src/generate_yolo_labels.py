"""
Auto-Label SUMO Screenshots for YOLO Fine-Tuning
=================================================
Captures screenshots from SUMO GUI and creates YOLO-format labels
using TraCI vehicle positions as ground truth.

Uses a dual-screenshot approach: takes a clean frame (training image),
then adds POI markers at vehicle positions, takes a second screenshot,
and detects marker pixel positions to generate pixel-perfect labels.

Usage:
    python generate_yolo_labels.py                    # all versions, 100 frames each
    python generate_yolo_labels.py --versions v3 v5   # specific versions
    python generate_yolo_labels.py --frames 200       # more frames per version
"""

import os
import sys
import math
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_simulation"
sys.path.insert(0, str(SCRIPT_DIR))

import traci

# ── Constants ──
WIDTH, HEIGHT = 1920, 1080
STEP_LENGTH = 1.0 / 30
CAPTURE_INTERVAL = 30  # capture every N sim frames (~1/sec)
# Marker color for POIs (bright magenta - rare in SUMO scenes)
MARKER_COLOR = (255, 0, 255, 255)  # RGBA


def _find_markers(clean_img, marker_img):
    """Find marker pixel positions by diffing clean vs marker screenshots.

    Returns list of (cx, cy) pixel coordinates where markers were detected.
    """
    # Convert to HSV and look for magenta markers
    hsv_m = cv2.cvtColor(marker_img, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)

    # Magenta in HSV: H around 150-170, high S and V
    mag_mask = cv2.inRange(hsv_m, (140, 80, 80), (175, 255, 255))
    mag_clean = cv2.inRange(hsv_c, (140, 80, 80), (175, 255, 255))

    # Markers = magenta in marker image but not in clean image
    diff = cv2.bitwise_and(mag_mask, cv2.bitwise_not(mag_clean))

    # Also try simple pixel difference for robustness
    abs_diff = cv2.absdiff(marker_img, clean_img)
    gray_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Combine both detection methods
    combined = cv2.bitwise_or(diff, thresh)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if 3 < area < 2000:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append((
                    M["m10"] / M["m00"],
                    M["m01"] / M["m00"],
                ))
    return centers


def generate_labels(versions, frames_per_version, output_dir, seeds=None):
    """Generate YOLO training data from SUMO simulations."""

    sumo_gui = shutil.which("sumo-gui")
    if not sumo_gui:
        raise RuntimeError("sumo-gui not found on PATH")

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    os.makedirs(str(images_dir / "train"), exist_ok=True)
    os.makedirs(str(images_dir / "val"), exist_ok=True)
    os.makedirs(str(labels_dir / "train"), exist_ok=True)
    os.makedirs(str(labels_dir / "val"), exist_ok=True)

    if seeds is None:
        seeds = [42, 43, 44]

    total_frames = 0
    total_vehicles = 0

    for version in versions:
        cfg = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
        view_xml = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.view.xml")

        if not os.path.exists(cfg):
            print(f"  [SKIP] No config for {version}")
            continue

        for seed in seeds:
            print(f"  [{version} seed={seed}] Capturing frames...")

            sumo_cmd = [
                sumo_gui, "-c", cfg,
                "--start", "--quit-on-end", "--gui-testing",
                "--window-size", f"{WIDTH},{HEIGHT}",
                "--delay", "0",
                "--step-length", str(STEP_LENGTH),
                "--no-warnings",
                "--seed", str(seed),
                "--end", "300",
            ]
            if os.path.exists(view_xml):
                sumo_cmd += ["--gui-settings-file", view_xml]

            traci.start(sumo_cmd)

            try:
                # Warm up + setup view
                for _ in range(20):
                    traci.simulationStep()
                traci.gui.setSchema("View #0", "real world")
                traci.gui.setZoom("View #0", 350)
                traci.gui.setOffset("View #0", 100, 100)
                for _ in range(5):
                    traci.simulationStep()

                frame_count = 0
                captured = 0
                clean_path = f"/tmp/sumo_clean_{version}_{seed}.png"
                marker_path = f"/tmp/sumo_marker_{version}_{seed}.png"

                while (traci.simulation.getMinExpectedNumber() > 0
                       and captured < frames_per_version):
                    traci.simulationStep()
                    frame_count += 1

                    if frame_count % CAPTURE_INTERVAL != 0:
                        continue

                    # Skip frames with few vehicles
                    veh_ids = traci.vehicle.getIDList()
                    if len(veh_ids) < 3:
                        continue

                    # ── Step 1: Clean screenshot (training image) ──
                    traci.gui.screenshot("View #0", clean_path,
                                         WIDTH, HEIGHT)
                    traci.simulationStep()
                    clean_img = cv2.imread(clean_path)
                    if clean_img is None:
                        continue

                    # ── Step 2: Get vehicle info and add markers ──
                    veh_info = {}
                    for vid in veh_ids:
                        try:
                            x, y = traci.vehicle.getPosition(vid)
                            length = traci.vehicle.getLength(vid)
                            width = traci.vehicle.getWidth(vid)
                            angle = traci.vehicle.getAngle(vid)
                            veh_info[vid] = (x, y, length, width, angle)
                            traci.poi.add(
                                f"_m_{vid}", x, y,
                                MARKER_COLOR, layer=200, width=3)
                        except traci.exceptions.TraCIException:
                            continue

                    if len(veh_info) < 3:
                        for vid in veh_info:
                            try:
                                traci.poi.remove(f"_m_{vid}")
                            except Exception:
                                pass
                        continue

                    # Let markers render
                    traci.simulationStep()

                    # ── Step 3: Marker screenshot ──
                    traci.gui.screenshot("View #0", marker_path,
                                         WIDTH, HEIGHT)
                    traci.simulationStep()
                    marker_img = cv2.imread(marker_path)

                    # Remove markers immediately
                    for vid in veh_info:
                        try:
                            traci.poi.remove(f"_m_{vid}")
                        except Exception:
                            pass

                    if marker_img is None:
                        continue

                    # ── Step 4: Find marker pixel positions ──
                    marker_centers = _find_markers(clean_img, marker_img)

                    if len(marker_centers) < 3:
                        continue

                    # ── Step 5: Compute bounding boxes ──
                    # We need to match marker centers to vehicles for
                    # size/angle info. Use the calibrated transform from
                    # marker positions to estimate scale.
                    # First, get scale from marker spread vs SUMO spread
                    sumo_xs = [v[0] for v in veh_info.values()]
                    sumo_ys = [v[1] for v in veh_info.values()]
                    px_xs = [c[0] for c in marker_centers]
                    px_ys = [c[1] for c in marker_centers]

                    sumo_range_x = max(sumo_xs) - min(sumo_xs)
                    sumo_range_y = max(sumo_ys) - min(sumo_ys)
                    px_range_x = max(px_xs) - min(px_xs)
                    px_range_y = max(px_ys) - min(px_ys)

                    if sumo_range_x > 5:
                        scale_x = px_range_x / sumo_range_x
                    else:
                        scale_x = 15.0  # fallback
                    if sumo_range_y > 5:
                        scale_y = px_range_y / sumo_range_y
                    else:
                        scale_y = 15.0
                    scale_px = (scale_x + scale_y) / 2

                    # ── Step 6: Generate YOLO labels ──
                    labels = []
                    for cx_px, cy_px in marker_centers:
                        # Skip markers outside frame (shouldn't happen)
                        if (cx_px < 2 or cx_px >= WIDTH - 2
                                or cy_px < 2 or cy_px >= HEIGHT - 2):
                            continue

                        # Find closest vehicle for dimensions
                        # (markers are at vehicle centers, so match by
                        # finding which SUMO vehicle maps closest)
                        best_vid = None
                        best_dist = float("inf")
                        for vid, (sx, sy, ln, wd, ang) in veh_info.items():
                            # Rough pixel estimate for matching
                            est_px = min(px_xs) + (sx - min(sumo_xs)) * scale_x
                            est_py = min(px_ys) + (max(sumo_ys) - sy) * scale_y
                            d = ((est_px - cx_px)**2
                                 + (est_py - cy_px)**2)**0.5
                            if d < best_dist:
                                best_dist = d
                                best_vid = vid

                        if best_vid is None:
                            # Use default vehicle size
                            bbox_w = 30
                            bbox_h = 55
                        else:
                            _, _, length, width, angle = veh_info[best_vid]
                            veh_len_px = length * scale_px
                            veh_wid_px = width * scale_px
                            angle_rad = math.radians(angle)
                            cos_a = abs(math.cos(angle_rad))
                            sin_a = abs(math.sin(angle_rad))
                            bbox_w = veh_len_px * sin_a + veh_wid_px * cos_a
                            bbox_h = veh_len_px * cos_a + veh_wid_px * sin_a

                        # Pad for sprite size
                        bbox_w *= 1.4
                        bbox_h *= 1.4

                        # Clamp to frame
                        x1 = max(0, cx_px - bbox_w / 2)
                        y1 = max(0, cy_px - bbox_h / 2)
                        x2 = min(WIDTH, cx_px + bbox_w / 2)
                        y2 = min(HEIGHT, cy_px + bbox_h / 2)

                        if (x2 - x1) < 5 or (y2 - y1) < 5:
                            continue

                        # YOLO format: class cx cy w h (normalized 0-1)
                        cx_norm = ((x1 + x2) / 2) / WIDTH
                        cy_norm = ((y1 + y2) / 2) / HEIGHT
                        w_norm = (x2 - x1) / WIDTH
                        h_norm = (y2 - y1) / HEIGHT

                        labels.append(f"0 {cx_norm:.6f} {cy_norm:.6f} "
                                      f"{w_norm:.6f} {h_norm:.6f}")

                    if not labels:
                        continue

                    # Split: 80% train, 20% val
                    split = "val" if random.random() < 0.2 else "train"
                    fname = f"{version}_{seed}_{captured:04d}"

                    # Save clean image (no markers)
                    dst_img = str(images_dir / split / f"{fname}.png")
                    if (clean_img.shape[1] != WIDTH
                            or clean_img.shape[0] != HEIGHT):
                        clean_img = cv2.resize(clean_img, (WIDTH, HEIGHT))
                    cv2.imwrite(dst_img, clean_img)

                    # Save label
                    dst_lbl = str(labels_dir / split / f"{fname}.txt")
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(labels))

                    total_vehicles += len(labels)
                    captured += 1
                    total_frames += 1

                print(f"    Captured {captured} frames")

            except traci.exceptions.FatalTraCIError:
                print(f"    SUMO crashed, got {captured} frames")
            finally:
                try:
                    traci.close()
                except Exception:
                    pass

            # Clean up temp files
            for p in [clean_path, marker_path]:
                if os.path.exists(p):
                    os.remove(p)

    # Write dataset YAML
    yaml_path = output_dir / "dataset.yaml"
    with open(str(yaml_path), "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['vehicle']\n")

    print(f"\n  Dataset generated:")
    print(f"    Total frames: {total_frames}")
    print(f"    Total vehicle labels: {total_vehicles}")
    print(f"    Avg vehicles/frame: {total_vehicles/max(total_frames,1):.1f}")

    # Count train/val split
    train_imgs = len(list((images_dir / "train").glob("*.png")))
    val_imgs = len(list((images_dir / "val").glob("*.png")))
    print(f"    Train: {train_imgs} images")
    print(f"    Val: {val_imgs} images")
    print(f"    YAML: {yaml_path}")

    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate YOLO training data from SUMO screenshots")
    parser.add_argument("--versions", nargs="+",
                        default=["v2", "v3", "v4", "v5", "v6", "v7"],
                        help="Scenario versions")
    parser.add_argument("--frames", type=int, default=100,
                        help="Frames to capture per version per seed (default: 100)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Random seeds (default: 42 43 44)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: data/sumo_yolo_dataset)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "sumo_yolo_dataset")

    print(f"{'='*60}")
    print(f"  AUTO-LABEL SUMO → YOLO DATASET")
    print(f"  Versions: {args.versions}")
    print(f"  Frames/version/seed: {args.frames}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    yaml_path = generate_labels(args.versions, args.frames, output_dir, args.seeds)

    print(f"\n  To fine-tune YOLO:")
    print(f"    yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=1280")
