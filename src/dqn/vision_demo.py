"""
Vision Demo — Run YOLO-based adaptive controller with full video recording
==========================================================================
Records a video showing:
  - YOLO bounding boxes + track IDs on every vehicle
  - Traffic light state (NS_GREEN / EW_GREEN / YELLOW)
  - Queue counts per direction (from YOLO detections)
  - Demand scores and green split decisions
  - Phase change markers

Usage:
    python vision_demo.py --version v3
    python vision_demo.py --version v3 --seed 42
    python vision_demo.py --version v3 --no-record   # watch live only
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent           # src/dqn/
PROJECT_ROOT = SCRIPT_DIR.parents[1]                   # Research_zhou/
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_simulation"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SUMO_DIR))

import traci
from queue_manager import QueueManager
from arrival_rate import ArrivalRateEstimator
from lstm_predictor import RealtimePredictor
from vision_detector import VisionDetector

# ── Constants ────────────────────────────────────────────────────────────
STEP_LENGTH = 1.0 / 30
FPS = 30
WIDTH, HEIGHT = 1920, 1080
CELL_SIZE = 50
MIN_GREEN = 10.0
MAX_GREEN = 60.0
YELLOW_DUR = 3.0

NS_GREEN_STATE  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW_STATE = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN_STATE  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW_STATE = "rrrrryyyyyrrrrryyyyy"

NS_DIRS = {"NB", "SB"}
EW_DIRS = {"EB", "WB"}

VISION_DETECT_INTERVAL = 15  # run YOLO every 15 frames

# Colors for drawing
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
BG_COLOR = (20, 20, 20)


def draw_zones(frame, queue_mgr, approach_to_dir, cell_size, alpha=0.15):
    """Draw intersection exclusion zone and approach ROIs on frame."""
    if queue_mgr is None:
        return frame

    overlay = frame.copy()

    # Approach ROI colors by direction
    dir_colors = {
        "NB": (0, 255, 128),   # green
        "SB": (0, 128, 255),   # blue
        "EB": (255, 128, 0),   # orange
        "WB": (255, 0, 255),   # magenta
    }

    # Draw approach ROIs
    for a in queue_mgr.approaches:
        label = a["label"]
        direction = approach_to_dir.get(label, "")
        color = dir_colors.get(direction, (200, 200, 200))
        for (cx, cy) in a["cells"]:
            px1 = int(cx * cell_size)
            py1 = int(cy * cell_size)
            cv2.rectangle(overlay, (px1, py1),
                          (px1 + cell_size, py1 + cell_size), color, -1)

    # Draw intersection exclusion zone
    m = queue_mgr.intersection_margin
    gw, gh = queue_mgr.grid_w, queue_mgr.grid_h
    ix1 = int(gw * m * cell_size)
    iy1 = int(gh * m * cell_size)
    ix2 = int(gw * (1 - m) * cell_size)
    iy2 = int(gh * (1 - m) * cell_size)
    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (0, 0, 200), -1)

    # Blend
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw borders and labels on top (not blended)
    # Intersection zone border + label
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 0, 200), 2)
    cv2.putText(frame, "EXCLUDED ZONE", (ix1 + 8, iy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    # Approach labels
    for a in queue_mgr.approaches:
        label = a["label"]
        direction = approach_to_dir.get(label, "?")
        cx, cy = a["center"]
        tx = int(cx * cell_size)
        ty = int(cy * cell_size)
        color = dir_colors.get(direction, (200, 200, 200))
        cv2.putText(frame, f"{label} ({direction})", (tx - 20, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def draw_overlay(frame, phase, phase_timer, ns_green_dur, ew_green_dur,
                 ns_queue, ew_queue, ns_score, ew_score,
                 detections, sim_time, decision_log_entry=None):
    """Draw all HUD overlays on a frame."""
    h, w = frame.shape[:2]

    # ── Traffic light state banner (top center) ──
    if "YELLOW" in phase:
        phase_color = YELLOW
        phase_label = f"YELLOW ({phase_timer:.1f}s / {YELLOW_DUR:.0f}s)"
    elif "NS" in phase:
        phase_color = GREEN
        phase_label = f"NS GREEN ({phase_timer:.1f}s / {ns_green_dur:.0f}s)"
    else:
        phase_color = GREEN
        phase_label = f"EW GREEN ({phase_timer:.1f}s / {ew_green_dur:.0f}s)"

    banner_w = 500
    banner_x = (w - banner_w) // 2
    cv2.rectangle(frame, (banner_x, 5), (banner_x + banner_w, 50), BG_COLOR, -1)
    cv2.rectangle(frame, (banner_x, 5), (banner_x + banner_w, 50), phase_color, 2)
    cv2.putText(frame, phase_label, (banner_x + 20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

    # ── Sim time (top left) ──
    cv2.rectangle(frame, (5, 5), (220, 50), BG_COLOR, -1)
    cv2.putText(frame, f"t = {sim_time:.1f}s", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    # ── Detection count (top right) ──
    n_det = len(detections)
    cv2.rectangle(frame, (w - 280, 5), (w - 5, 50), BG_COLOR, -1)
    cv2.putText(frame, f"YOLO: {n_det} vehicles", (w - 270, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

    # ── Queue & Score panel (left side) ──
    panel_x, panel_y = 5, 60
    panel_w, panel_h = 320, 180
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), BG_COLOR, -1)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), WHITE, 1)

    y = panel_y + 25
    cv2.putText(frame, "VISION-BASED CONTROL", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2)
    y += 30
    cv2.putText(frame, f"NS queue: {ns_queue}", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if "NS" in phase and "YELLOW" not in phase else WHITE, 1)
    y += 25
    cv2.putText(frame, f"EW queue: {ew_queue}", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if "EW" in phase and "YELLOW" not in phase else WHITE, 1)
    y += 30
    cv2.putText(frame, f"NS score: {ns_score:.1f}", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
    y += 25
    cv2.putText(frame, f"EW score: {ew_score:.1f}", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
    y += 25
    cv2.putText(frame, f"Green: NS={ns_green_dur:.0f}s  EW={ew_green_dur:.0f}s",
                (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    # ── Decision log (bottom left) ──
    if decision_log_entry:
        dy = h - 50
        cv2.rectangle(frame, (5, dy - 5), (600, h - 5), BG_COLOR, -1)
        cv2.putText(frame, f"DECISION: {decision_log_entry}",
                    (15, dy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

    # ── Draw YOLO bounding boxes ──
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        tid = det["track_id"]
        conf = det["conf"]

        # Box color: green if moving, red if likely queued (small box = stopped)
        box_w = x2 - x1
        box_h = y2 - y1
        box_color = GREEN

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"#{tid} {conf:.0%}"
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return frame


def run_vision_demo(version, seed=42, record=True):
    """Run vision-based adaptive controller with full video recording."""

    # ── Load models ──
    yolo_path = str(PROJECT_ROOT / "models" / "yolov8n-visdrone.pt")
    detector = VisionDetector(yolo_path)

    lstm_path = str(PROJECT_ROOT / "models" / f"fixed_{version}_trajectories_lstm_model.pt")
    lstm = RealtimePredictor(lstm_path) if os.path.exists(lstm_path) else None

    traj_path = str(PROJECT_ROOT / "data" / "transitions" / "sumo_transition"
                     / f"fixed_{version}_trajectories.json")
    traj_data = None
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            traj_data = json.load(f)

    # Load optimized params
    opt_path = SUMO_DIR / "optimized_params.json"
    if os.path.exists(str(opt_path)):
        with open(str(opt_path)) as f:
            opt = json.load(f)
        if opt.get("mode") == "per_version" and version in opt.get("per_version", {}):
            params = opt["per_version"][version]["best_params"]
        else:
            params = {"w_queue": 1.0, "w_arrival": 3.0, "w_through": 0.3, "cycle_total": 90.0}
    else:
        params = {"w_queue": 1.0, "w_arrival": 3.0, "w_through": 0.3, "cycle_total": 90.0}

    # ── Output directory ──
    out_dir = SCRIPT_DIR / f"vision_demo/{version}"
    os.makedirs(str(out_dir), exist_ok=True)
    output_video = str(out_dir / "vision_adaptive.mp4")

    # ── Launch sumo-gui ──
    sumo_gui = shutil.which("sumo-gui")
    assert sumo_gui, "sumo-gui not found on PATH"

    cfg = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
    view_xml = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.view.xml")

    sumo_cmd = [
        sumo_gui, "-c", cfg,
        "--start", "--quit-on-end", "--gui-testing",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0",
        "--step-length", str(STEP_LENGTH),
        "--no-warnings",
        "--waiting-time-memory", "3600",
        "--seed", str(seed), "--end", "300", "--time-to-teleport", "120",
    ]
    if os.path.exists(view_xml):
        sumo_cmd += ["--gui-settings-file", view_xml]

    print(f"\n[Vision Demo {version}] Launching sumo-gui...")
    traci.start(sumo_cmd)

    try:
        # Warm up + GUI setup
        for _ in range(15):
            traci.simulationStep()
        traci.gui.setSchema("View #0", "real world")
        traci.gui.setZoom("View #0", 350)
        for _ in range(5):
            traci.simulationStep()

        # ── Setup QueueManager ──
        grid_w = WIDTH // CELL_SIZE + 1
        grid_h = HEIGHT // CELL_SIZE + 1

        queue_mgr = None
        approach_to_dir = {}
        if traj_data:
            queue_mgr = QueueManager(grid_w, grid_h, fps=FPS,
                                     intersection_dist=30.0, intersection_margin=0.35)
            queue_mgr.auto_discover_approaches(traj_data["trajectories"],
                                               traj_data["grid_cols"])
            queue_mgr.calibrate_with_cell_size(CELL_SIZE)

            gw_half = grid_w / 2
            gh_half = grid_h / 2
            for a in queue_mgr.approaches:
                cx, cy = a["center"]
                dx, dy = abs(cx - gw_half), abs(cy - gh_half)
                if dy > dx:
                    approach_to_dir[a["label"]] = "NB" if cy < gh_half else "SB"
                else:
                    approach_to_dir[a["label"]] = "WB" if cx < gw_half else "EB"

        arrival_est = ArrivalRateEstimator(fps=FPS, log_enabled=False)

        # ── Import scoring function ──
        from optimize_weights import compute_demand_score, compute_green_split, compute_scores

        # Take over TLS
        traci.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)

        # ── State ──
        current_phase = "NS_GREEN"
        phase_timer = 0.0
        ns_green_dur = ew_green_dur = params["cycle_total"] / 2

        vehicle_cells = defaultdict(list)
        vehicle_entry_dir = {}
        prev_cells = {}
        prev_vids = set()
        frame_count = 0

        last_detections = {}
        last_raw_detections = []  # for drawing bboxes
        ns_queue = ew_queue = 0
        ns_score = ew_score = 0.0

        # Metrics
        veh_last_wait = {}
        veh_total_wait = []
        departed = arrived = 0

        # Video writer
        tmp_screenshot = tempfile.mktemp(suffix=".png")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, FPS, (WIDTH, HEIGHT)) if record else None
        written = 0

        # Decision log
        decisions = []
        last_decision_text = None

        print(f"[Vision Demo {version}] Running vision-based adaptive controller...")
        print(f"  Params: Q={params['w_queue']:.1f} A={params['w_arrival']:.1f} "
              f"T={params['w_through']:.2f} C={params['cycle_total']:.0f}s")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            phase_timer += STEP_LENGTH
            frame_count += 1
            sim_time = traci.simulation.getTime()

            # ── YOLO detection ──
            if frame_count % VISION_DETECT_INTERVAL == 0:
                raw_dets = detector.detect(traci, width=WIDTH, height=HEIGHT)
                last_raw_detections = raw_dets
                new_detections = {}
                for d in raw_dets:
                    tid = d["track_id"]
                    px, py = d["center"]
                    cell = (int(px // CELL_SIZE), int(py // CELL_SIZE))
                    new_detections[tid] = {"center": (px, py), "cell": cell}
                last_detections = new_detections

            # ── Vision-based vehicle tracking ──
            current_vids = set()
            active_cells = {}

            for tid, det in last_detections.items():
                current_vids.add(tid)
                px, py = det["center"]
                cell = det["cell"]
                active_cells[tid] = cell

                if queue_mgr:
                    had = tid in queue_mgr.vehicle_approach
                    queue_mgr.track_vehicle(tid, cell, prev_cells.get(tid),
                                            pixel_pos=(px, py))
                    if not had and tid in queue_mgr.vehicle_approach:
                        arrival_est.record_arrival(
                            tid, queue_mgr.vehicle_approach[tid], frame_count)
                        approach_label = queue_mgr.vehicle_approach[tid]
                        if approach_label in approach_to_dir:
                            vehicle_entry_dir[tid] = approach_to_dir[approach_label]

                if tid not in prev_cells or prev_cells[tid] != cell:
                    vehicle_cells[tid].append(cell)
                prev_cells[tid] = cell

            for tid in prev_vids - current_vids:
                if queue_mgr:
                    queue_mgr.vehicle_exited(tid)
                arrival_est.vehicle_exited(tid)
            prev_vids = current_vids

            # ── Metrics (TraCI ground truth) ──
            for vid in traci.vehicle.getIDList():
                veh_last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
            for vid in traci.simulation.getArrivedIDList():
                if vid in veh_last_wait:
                    veh_total_wait.append(veh_last_wait.pop(vid))
            departed += traci.simulation.getDepartedNumber()
            arrived += traci.simulation.getArrivedNumber()

            # ── Compute queue counts for display ──
            if queue_mgr and frame_count % VISION_DETECT_INTERVAL == 0:
                ns_queue = ew_queue = 0
                state = queue_mgr.compute_state(active_cells)
                for label, data in state.items():
                    d = approach_to_dir.get(label)
                    if d in NS_DIRS:
                        ns_queue += data["queue"]
                    elif d in EW_DIRS:
                        ew_queue += data["queue"]

            # ── Phase transitions ──
            phase_changed = False
            if current_phase == "NS_GREEN" and phase_timer >= ns_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
                current_phase = "NS_YELLOW"
                phase_timer = 0.0
                phase_changed = True

            elif current_phase == "NS_YELLOW" and phase_timer >= YELLOW_DUR:
                ns_score, ew_score, ns_green_dur, ew_green_dur = compute_scores(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, params)
                traci.trafficlight.setRedYellowGreenState("C", EW_GREEN_STATE)
                current_phase = "EW_GREEN"
                phase_timer = 0.0
                phase_changed = True
                last_decision_text = (f"Switch to EW | NS_score={ns_score:.1f} "
                                      f"EW_score={ew_score:.1f} | "
                                      f"NS={ns_green_dur:.0f}s EW={ew_green_dur:.0f}s")
                decisions.append({
                    "time": round(sim_time, 1), "action": "SWITCH_TO_EW",
                    "ns_score": round(ns_score, 1), "ew_score": round(ew_score, 1),
                    "ns_green": round(ns_green_dur, 1), "ew_green": round(ew_green_dur, 1),
                    "ns_queue": ns_queue, "ew_queue": ew_queue,
                })

            elif current_phase == "EW_GREEN" and phase_timer >= ew_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
                current_phase = "EW_YELLOW"
                phase_timer = 0.0
                phase_changed = True

            elif current_phase == "EW_YELLOW" and phase_timer >= YELLOW_DUR:
                ns_score, ew_score, ns_green_dur, ew_green_dur = compute_scores(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, params)
                traci.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
                current_phase = "NS_GREEN"
                phase_timer = 0.0
                phase_changed = True
                last_decision_text = (f"Switch to NS | NS_score={ns_score:.1f} "
                                      f"EW_score={ew_score:.1f} | "
                                      f"NS={ns_green_dur:.0f}s EW={ew_green_dur:.0f}s")
                decisions.append({
                    "time": round(sim_time, 1), "action": "SWITCH_TO_NS",
                    "ns_score": round(ns_score, 1), "ew_score": round(ew_score, 1),
                    "ns_green": round(ns_green_dur, 1), "ew_green": round(ew_green_dur, 1),
                    "ns_queue": ns_queue, "ew_queue": ew_queue,
                })

            # ── Capture frame + draw overlays ──
            if record:
                traci.gui.screenshot("View #0", tmp_screenshot, WIDTH, HEIGHT)
                frame = cv2.imread(tmp_screenshot)
                if frame is not None:
                    if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                        frame = cv2.resize(frame, (WIDTH, HEIGHT))
                    # Draw zones first (semi-transparent, under everything)
                    frame = draw_zones(frame, queue_mgr, approach_to_dir, CELL_SIZE)
                    # Then draw HUD + YOLO boxes on top
                    frame = draw_overlay(
                        frame, current_phase, phase_timer,
                        ns_green_dur, ew_green_dur,
                        ns_queue, ew_queue, ns_score, ew_score,
                        last_raw_detections, sim_time,
                        last_decision_text if phase_changed else None,
                    )
                    writer.write(frame)
                    written += 1

            # Progress
            if frame_count % (FPS * 10) == 0:
                print(f"  t={sim_time:.0f}s | phase={current_phase} | "
                      f"YOLO={len(last_raw_detections)} vehicles | "
                      f"NS_q={ns_queue} EW_q={ew_queue} | frames={written}")
                sys.stdout.flush()

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try:
            traci.simulationStep()
        except Exception:
            pass
        traci.close()

    # Cleanup
    if os.path.exists(tmp_screenshot):
        os.remove(tmp_screenshot)
    detector.cleanup()

    if writer:
        writer.release()

    # Survivorship bias fix
    veh_total_wait.extend(veh_last_wait.values())
    arrived += len(veh_last_wait)
    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    avg_wait = total_wait / max(arrived, 1)

    # Save decision log
    log_path = str(out_dir / "decision_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "version": version, "seed": seed,
            "params": params,
            "avg_wait": round(avg_wait, 2),
            "decisions": decisions,
        }, f, indent=2)

    print(f"\n[Vision Demo {version}] Done!")
    print(f"  Avg wait: {avg_wait:.2f}s")
    print(f"  Video: {written} frames ({written/FPS:.1f}s) → {output_video}")
    print(f"  Decisions: {len(decisions)} phase changes → {log_path}")

    return avg_wait


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision-based adaptive controller demo with YOLO overlays")
    parser.add_argument("--version", default="v3",
                        help="Scenario version (default: v3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-record", action="store_true",
                        help="Watch live without recording")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  VISION-BASED ADAPTIVE DEMO")
    print(f"  Version: {args.version}  Seed: {args.seed}")
    print(f"  YOLO model: yolov8n-visdrone.pt")
    print(f"  Recording: {'OFF' if args.no_record else 'ON'}")
    print(f"{'='*60}")

    run_vision_demo(args.version, seed=args.seed, record=not args.no_record)
