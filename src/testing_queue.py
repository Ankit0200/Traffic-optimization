"""
Queue-Only Testing
===================
Usage:
    python testing_queue.py --video data.mov --model model.pt --yolo yolo.pt --trajectories traj.json

Shows only queue detection and approach assignments — no exit predictions on vehicles.
Focused view for debugging/tuning queue manager behavior.

Controls:
    SPACE : Pause / Resume
    'q'   : Quit
    'g'   : Toggle grid
    'r'   : Toggle ROI overlay
"""

import cv2
import json
import argparse
from collections import defaultdict

from ultralytics import YOLO
from queue_manager import QueueManager
from arrival_rate import ArrivalRateEstimator


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
            cv2.putText(frame, f"{cx},{cy}", (x+2, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Queue-Only Testing")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="LSTM model (for cell_size)")
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--cell_size", type=int, default=50)
    parser.add_argument("--output", help="Save output video")
    parser.add_argument("--log", action="store_true", help="Enable CSV arrival logging")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    args = parser.parse_args()

    # Get cell_size from model
    import torch
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    model_cell_size = ckpt["cell_size"]
    print(f"Model cell_size: {model_cell_size}")

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

    # Queue Manager Setup
    grid_w = frame_w // model_cell_size + 1
    grid_h = frame_h // model_cell_size + 1
    queue_mgr = QueueManager(grid_w, grid_h)

    print("\n── Setting up queue manager ──")
    with open(args.trajectories) as f:
        traj_data = json.load(f)
    grid_cols = traj_data["grid_cols"]
    trajectories = traj_data["trajectories"]

    print("  Auto-discovering approaches...")
    queue_mgr.auto_discover_approaches(trajectories, grid_cols)
    queue_mgr.calibrate_with_cell_size(model_cell_size)

    # Arrival Rate Estimator
    arrival_est = ArrivalRateEstimator(fps=fps, log_enabled=args.log)

    prev_cells = {}
    prev_frame_ids = set()
    frame_number = 0
    total_exited = 0

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
            frame = draw_grid(frame, model_cell_size)

        if show_rois:
            frame = queue_mgr.draw_approach_rois(frame, model_cell_size)

        # YOLO tracking
        results = yolo.track(frame, persist=True, classes=[3, 4, 5, 8], conf=0.15, imgsz=args.imgsz)

        active_cells = {}

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

                had_approach = tid in queue_mgr.vehicle_approach
                queue_mgr.track_vehicle(tid, current_cell, prev_cells.get(tid), pixel_pos=(cx, cy))
                if not had_approach and tid in queue_mgr.vehicle_approach:
                    arrival_est.record_arrival(tid, queue_mgr.vehicle_approach[tid], frame_number)
                prev_cells[tid] = current_cell

                # Draw vehicle — queue-only view
                queued = queue_mgr.is_queued(tid)
                approach = queue_mgr.vehicle_approach.get(tid, "")

                # Color by approach
                approach_color = queue_mgr._get_color(approach) if approach else (128, 128, 128)

                vel = queue_mgr.get_velocity(tid)

                if queued:
                    # Filled overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), approach_color, -1)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    # Thick border
                    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), approach_color, 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    # Badge
                    badge_text = f"QUEUED [{approach}]"
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - 28 - th), (x1 + tw + 8, y1 - 24), approach_color, -1)
                    cv2.putText(frame, badge_text, (x1 + 4, y1 - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    # ID label with velocity
                    cv2.putText(frame, f"ID:{tid} [{approach}|Q {vel:.1f}m/s]",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, approach_color, 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), approach_color, 2)
                    if approach:
                        label = f"ID:{tid} [{approach} {vel:.1f}m/s]"
                    else:
                        label = f"ID:{tid} [{vel:.1f}m/s]"
                    cv2.putText(frame, label,
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, approach_color, 1)

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Compute queue state
        queue_state = queue_mgr.compute_state(active_cells)

        # Detect disappeared tracks
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            queue_mgr.vehicle_exited(tid)
            arrival_est.vehicle_exited(tid)
            total_exited += 1

        prev_frame_ids = current_frame_ids

        # Compute arrival rates
        arrival_rates = arrival_est.compute_all_rates(frame_number)

        # HUD — queue state panel
        frame = queue_mgr.draw_state(frame, queue_state, model_cell_size)

        # HUD — arrival rate panel
        ar_px = 10
        ar_py = 80
        cv2.putText(frame, "ARRIVAL RATE", (ar_px, ar_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        ar_y = ar_py + 22
        for approach in sorted(arrival_rates):
            rate = arrival_rates[approach]
            color = queue_mgr._get_color(approach) if approach else (180, 180, 180)
            cv2.putText(frame, f"{approach}: {rate:.2f} veh/s",
                        (ar_px, ar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            ar_y += 20

        # Info bar
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames} | "
                    f"Active: {len(current_frame_ids)} | "
                    f"Exited: {total_exited}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Grid: {'ON' if show_grid else 'OFF'} (g) | "
                    f"ROIs: {'ON' if show_rois else 'OFF'} (r) | "
                    f"{'PAUSED' if paused else 'PLAYING'} (space)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if writer:
            writer.write(frame)
            if frame_number % 30 == 0:
                print(f"  Processed frame {frame_number}/{total_frames}...")
        else:
            cv2.imshow("Queue Testing", frame)

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
    arrival_est.close()
    if writer:
        writer.release()
    if not args.output:
        cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  QUEUE TESTING SUMMARY")
    print(f"  Total vehicles tracked: {total_exited}")
    print(f"  Queue manager completed: {queue_mgr.total_completed}")
    print(f"  Approaches: {len(queue_mgr.approaches)}")
    for a in queue_mgr.approaches:
        print(f"    {a['label']}: center=({a['center'][0]:.1f}, {a['center'][1]:.1f}), "
              f"{a['count']} traj, {len(a['cells'])} ROI cells")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
