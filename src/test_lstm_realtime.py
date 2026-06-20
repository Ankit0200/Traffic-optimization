"""
Real-Time LSTM Turn Prediction
================================
Usage:
    python test_lstm_realtime.py --video control.mp4 --model control_trajectories_lstm_model.pt

Runs YOLO tracking on video and uses the trained LSTM to predict
each vehicle's exit direction in real-time as it moves.

Controls:
    SPACE : Pause / Resume
    'q'   : Quit
    'g'   : Toggle grid
"""

import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
from queue_manager import QueueManager
from lstm_predictor import RealtimePredictor


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_cell(x, y, cell_size):
    return (int(x // cell_size), int(y // cell_size))


def draw_grid(frame, cell_size, color=(100, 100, 100), thickness=1):
    h, w = frame.shape[:2]
    for x in range(0, w, cell_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    for y in range(0, h, cell_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cx, cy = x // cell_size, y // cell_size
            cv2.putText(frame, f"{cx},{cy}", (x+2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)

    return frame


# Color per exit zone
EXIT_COLORS = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
]


def draw_exit_zones(frame, clusters, label_map, display_cell_size, model_cell_size, alpha=0.45):
    """Draw highlighted colored rectangles on exit zone cells with borders."""
    overlay = frame.copy()

    for cl in clusters:
        label = cl["label"]
        idx = label_map.get(label, 0)
        color = EXIT_COLORS[idx % len(EXIT_COLORS)]

        cells = cl["cells"]

        for c in cells:
            cx_orig, cy_orig = c[0], c[1]
            px1 = int(cx_orig * model_cell_size)
            py1 = int(cy_orig * model_cell_size)
            px2 = px1 + model_cell_size
            py2 = py1 + model_cell_size

            cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        center_x = int(cl["center"][0] * model_cell_size + model_cell_size // 2)
        center_y = int(cl["center"][1] * model_cell_size + model_cell_size // 2)
        text = label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(overlay, (center_x - 22, center_y - th - 4),
                      (center_x + tw - 16, center_y + 4), (0, 0, 0), -1)
        cv2.putText(overlay, text, (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Run on video
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real-Time LSTM Turn Prediction")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to trained LSTM model (.pt)")
    parser.add_argument("--yolo", default="../models/10_epoch.pt", help="YOLO model path")
    parser.add_argument("--cell_size", type=int, default=50, help="Grid cell size")
    parser.add_argument("--output", help="Path to save output video (e.g., output.mp4)")
    parser.add_argument("--trajectories", help="Path to trajectories JSON (for queue manager bootstrap)")
    parser.add_argument("--signal_log", help="Path to decision_log.json for signal phase overlay")
    args = parser.parse_args()

    # Load predictor (LSTM — independent exit prediction)
    print("\n── Loading LSTM model ──")
    predictor = RealtimePredictor(args.model)

    # Load YOLO
    print("\n── Loading YOLO ──")
    yolo = YOLO(args.yolo)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.video}'")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} FPS, {total_frames} frames")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))

    # Tracking state
    display_cell_size = args.cell_size
    model_cell_size = predictor.cell_size

    # ── Queue Manager Setup (approach-only, no exit coupling) ─────────
    grid_w = frame_w // model_cell_size + 1
    grid_h = frame_h // model_cell_size + 1
    queue_mgr = QueueManager(grid_w, grid_h, fps=fps)

    if args.trajectories:
        print("\n── Setting up queue manager ──")
        with open(args.trajectories) as f:
            traj_data = json.load(f)
        grid_cols = traj_data["grid_cols"]
        trajectories = traj_data["trajectories"]

        print("  Auto-discovering approach ROIs...")
        queue_mgr.auto_discover_approaches(trajectories, grid_cols)
        queue_mgr.calibrate_with_cell_size(model_cell_size)
    else:
        print("\n  No --trajectories provided, queue manager will learn from scratch")

    # ── Signal phase overlay setup ─────────────────────────────────────
    signal_cycles = []
    if args.signal_log:
        with open(args.signal_log) as f:
            signal_cycles = json.load(f)
        print(f"\n── Signal log loaded: {len(signal_cycles)} cycles ──")

    # Build phase timeline from decision log
    # Each cycle alternates: odd cycles → EW_GREEN, even cycles → NS_GREEN
    # Yellow duration = 3s between phases
    YELLOW_DUR = 3.0
    phase_events = []  # [(start_time, end_time, phase_name, cycle_data)]
    if signal_cycles:
        # Before first cycle: NS_GREEN from t=0 to first cycle time
        phase_events.append((0, signal_cycles[0]["time"] - YELLOW_DUR, "NS_GREEN", None))
        phase_events.append((signal_cycles[0]["time"] - YELLOW_DUR, signal_cycles[0]["time"], "NS_YELLOW", None))

        for i, cyc in enumerate(signal_cycles):
            is_ew = (i % 2 == 0)  # odd cycles (1,3,5..) switch TO EW, even (2,4,6..) switch TO NS
            phase = "EW_GREEN" if is_ew else "NS_GREEN"
            green_dur = cyc["ew_green"] if is_ew else cyc["ns_green"]

            end_green = cyc["time"] + green_dur
            if i + 1 < len(signal_cycles):
                next_t = signal_cycles[i + 1]["time"]
                phase_events.append((cyc["time"], next_t - YELLOW_DUR, phase, cyc))
                yellow_phase = "EW_YELLOW" if is_ew else "NS_YELLOW"
                phase_events.append((next_t - YELLOW_DUR, next_t, yellow_phase, None))
            else:
                phase_events.append((cyc["time"], cyc["time"] + green_dur, phase, cyc))

    def get_phase_info(sim_time):
        """Return (phase_name, cycle_data, time_remaining) for given sim time."""
        for start, end, phase, cyc in phase_events:
            if start <= sim_time < end:
                return phase, cyc, end - sim_time
        return "NS_GREEN", None, 0

    # {track_id: [(cx, cy), ...]} — unique cells per vehicle
    vehicle_cells = defaultdict(list)
    prev_cells = {}
    prev_frame_ids = set()
    frame_number = 0

    # LSTM predictions per vehicle (independent from queue)
    predictions = {}

    # Stats
    stats = {"total_predicted": 0, "total_exited_labeled": 0}

    show_grid = False
    show_rois = True
    paused = False

    print(f"\nRunning... SPACE pause | 'q' quit | 'g' grid | 'r' ROIs\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_frame_ids = set()

        if show_grid:
            frame = draw_grid(frame, display_cell_size)

        # Draw approach ROIs and exit zones
        if show_rois:
            frame = queue_mgr.draw_approach_rois(frame, model_cell_size)
        frame = draw_exit_zones(frame, predictor.clusters, predictor.label_map, display_cell_size, model_cell_size)

        # YOLO tracking
        results = yolo.track(frame, persist=True, classes=[3, 4, 5, 8], conf=0.15)

        active_cells = {}  # {tid: (cx, cy)} for queue state

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_cell = pixel_to_cell(cx, cy, model_cell_size)
                current_frame_ids.add(tid)
                active_cells[tid] = current_cell

                # Update queue manager (approach assignment + velocity-based stop detection)
                queue_mgr.track_vehicle(tid, current_cell, prev_cells.get(tid),
                                        pixel_pos=(cx, cy))

                # Record cell (only if different from last)
                if tid not in prev_cells or prev_cells[tid] != current_cell:
                    vehicle_cells[tid].append(current_cell)
                prev_cells[tid] = current_cell

                # Run LSTM prediction (independent from queue)
                cells_seq = vehicle_cells[tid]
                pred_label, confidence, all_probs = predictor.predict(cells_seq)

                if pred_label is not None:
                    predictions[tid] = {
                        "label": pred_label,
                        "confidence": confidence,
                        "probs": all_probs,
                        "steps": len(cells_seq)
                    }

                # Draw vehicle
                queued = queue_mgr.is_queued(tid)
                approach = queue_mgr.vehicle_approach.get(tid, "")

                if tid in predictions:
                    pred = predictions[tid]
                    label_idx = predictor.label_map.get(pred["label"], 0)
                    color = EXIT_COLORS[label_idx % len(EXIT_COLORS)]
                    conf = pred["confidence"]
                    steps_text = "prior" if pred["steps"] < 3 else f"{pred['steps']} steps"

                    if queued:
                        # Filled semi-transparent overlay to make queued vehicles pop
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                        # Thick colored border
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color, 3)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                        # "QUEUED" badge above the box
                        badge_text = f"QUEUED [{approach}]"
                        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - 42 - th), (x1 + tw + 8, y1 - 38), color, -1)
                        cv2.putText(frame, badge_text, (x1 + 4, y1 - 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    vel = queue_mgr.get_velocity(tid)
                    status = f"[{approach}|Q {vel:.1f}m/s]" if queued else f"[{approach} {vel:.1f}m/s]" if approach else f"[{vel:.1f}m/s]"
                    cv2.putText(frame, f"ID:{tid} {pred['label']} {status}",
                                (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                    cv2.putText(frame, f"{conf:.0%} ({steps_text})",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (200, 200, 200), 1)

                    bar_w = int(conf * 60)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 12), color, -1)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + 60, y2 + 12), color, 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, f"ID:{tid} (waiting...)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (128, 128, 128), 1)

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Compute queue state (approach queues only)
        queue_state = queue_mgr.compute_state(active_cells)

        # Detect disappeared tracks
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            queue_mgr.vehicle_exited(tid)
            if tid in predictions:
                pred = predictions[tid]
                stats["total_predicted"] += 1
                print(f"  Vehicle {tid} exited → predicted: {pred['label']} "
                      f"({pred['confidence']:.0%}, {pred['steps']} steps)")

        prev_frame_ids = current_frame_ids

        # ── HUD ───────────────────────────────────────────────────────
        # Queue state panel (with LSTM exit breakdown)
        frame = queue_mgr.draw_state(frame, queue_state, model_cell_size,
                                     exit_predictions=predictions)

        # Legend
        y_offset = 80
        for label, idx in predictor.label_map.items():
            color = EXIT_COLORS[idx % len(EXIT_COLORS)]
            cv2.rectangle(frame, (10, y_offset), (25, y_offset + 12), color, -1)
            cv2.putText(frame, label, (30, y_offset + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18

        # Active predictions count
        active_preds = sum(1 for tid in current_frame_ids if tid in predictions)
        sim_time = frame_number / fps
        hud_x = frame_w - 700
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames} | "
                    f"t={sim_time:.1f}s | "
                    f"Active: {len(current_frame_ids)} | "
                    f"Predicting: {active_preds} | "
                    f"Completed: {stats['total_predicted']}",
                    (hud_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Grid: {'ON' if show_grid else 'OFF'} (g) | "
                    f"ROIs: {'ON' if show_rois else 'OFF'} (r) | "
                    f"{'PAUSED' if paused else 'PLAYING'} (space)",
                    (hud_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Signal Phase Overlay ──────────────────────────────────────
        if signal_cycles:
            phase, cyc_data, time_left = get_phase_info(sim_time)

            # Phase indicator — top left
            panel_x = 20
            panel_y = 10

            # Background panel
            overlay_panel = frame.copy()
            panel_h = 160 if cyc_data else 80
            cv2.rectangle(overlay_panel, (panel_x - 10, panel_y - 5),
                          (panel_x + 410, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay_panel, 0.7, frame, 0.3, 0, frame)

            # Phase color
            if "NS" in phase and "YELLOW" not in phase:
                phase_color = (0, 255, 0)  # green
                phase_label = "NS GREEN"
            elif "EW" in phase and "YELLOW" not in phase:
                phase_color = (0, 200, 255)  # orange-ish
                phase_label = "EW GREEN"
            elif "YELLOW" in phase:
                phase_color = (0, 255, 255)  # yellow
                phase_label = phase.replace("_", " ")
            else:
                phase_color = (200, 200, 200)
                phase_label = phase

            # Traffic light icon
            cv2.circle(frame, (panel_x + 10, panel_y + 18), 12, phase_color, -1)
            cv2.circle(frame, (panel_x + 10, panel_y + 18), 13, (255, 255, 255), 2)

            cv2.putText(frame, f"{phase_label}  ({time_left:.1f}s left)",
                        (panel_x + 30, panel_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2)

            # Progress bar for phase timer
            bar_x = panel_x
            bar_y = panel_y + 38
            bar_w = 390
            if cyc_data:
                is_ew = "EW" in phase
                total_dur = cyc_data["ew_green"] if is_ew else cyc_data["ns_green"]
                elapsed = total_dur - time_left
                progress = min(1.0, elapsed / total_dur) if total_dur > 0 else 0
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (60, 60, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 8), phase_color, -1)

            # Scoring details when we have cycle data
            if cyc_data:
                y = panel_y + 58
                ns_s = cyc_data["ns_score"]
                ew_s = cyc_data["ew_score"]
                ns_g = cyc_data["ns_green"]
                ew_g = cyc_data["ew_green"]
                total_s = ns_s + ew_s if (ns_s + ew_s) > 0 else 1

                cv2.putText(frame, f"Cycle {cyc_data['cycle']}  |  {cyc_data['vehicles']} vehicles",
                            (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 22

                # NS score bar
                ns_pct = ns_s / total_s
                cv2.putText(frame, f"NS: {ns_s:.1f}", (panel_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.rectangle(frame, (panel_x + 80, y - 10), (panel_x + 80 + int(200 * ns_pct), y), (0, 255, 0), -1)
                cv2.putText(frame, f"{ns_g:.0f}s", (panel_x + 290, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y += 22

                # EW score bar
                ew_pct = ew_s / total_s
                cv2.putText(frame, f"EW: {ew_s:.1f}", (panel_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                cv2.rectangle(frame, (panel_x + 80, y - 10), (panel_x + 80 + int(200 * ew_pct), y), (0, 200, 255), -1)
                cv2.putText(frame, f"{ew_g:.0f}s", (panel_x + 290, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                y += 22

                cv2.putText(frame, f"Green: NS={ns_g:.0f}s  EW={ew_g:.0f}s  (budget 90s)",
                            (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if writer:
            writer.write(frame)
            if frame_number % 30 == 0:
                print(f"  Processed frame {frame_number}/{total_frames}...")
        else:
            cv2.imshow("LSTM Turn Prediction", frame)

            wait_time = int(1000 / fps) if writer is None else 1
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                while paused:
                    k2 = cv2.waitKey(30) & 0xFF
                    if k2 == ord(' '):
                        paused = False
                        break
                    elif k2 == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif k2 == ord('g'):
                        show_grid = not show_grid
                    elif k2 == ord('r'):
                        show_rois = not show_rois
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('r'):
                show_rois = not show_rois

    cap.release()
    if writer:
        writer.release()
    if not args.output:
        cv2.destroyAllWindows()

    # Final summary
    print(f"\n{'='*50}")
    print(f"  SESSION SUMMARY")
    print(f"  Total vehicles predicted: {stats['total_predicted']}")
    print(f"  Total vehicles tracked:   {len(vehicle_cells)}")
    print(f"  Queue manager completed:  {queue_mgr.total_completed}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
