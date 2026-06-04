"""
SUMO GUI Video Capture — Adaptive Timer v1 (Full Pipeline)
===========================================================
Same intersection and traffic as fixed_timer/v1, but signal timing
is controlled adaptively using the full research pipeline:

    1. QueueManager   — velocity-based queue detection per approach
    2. ArrivalRate    — rolling-window λ per approach (vehicles/sec)
    3. LSTM Predictor — turn intention prediction from partial trajectories
    4. Signal Optimizer — combines all three signals to allocate green time

The LSTM predictions tell us HOW vehicles will move (through vs turn),
arrival rates tell us the TREND of demand, and queue lengths tell us
the CURRENT state. Together they produce smarter green splits than
any single signal alone.

Usage:
    python run_sumo_video.py
"""

import os
import sys
import time
import shutil
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

import traci
import cv2
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# ── Project paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # data/sumo_simulation/adaptive_timer/v1 → project root
SRC_DIR = PROJECT_ROOT / "src"

# Add src/ to path for imports
sys.path.insert(0, str(SRC_DIR))

from queue_manager import QueueManager
from arrival_rate import ArrivalRateEstimator

SUMO_CFG = str(SCRIPT_DIR / "intersection.sumocfg")
OUTPUT_VIDEO = str(SCRIPT_DIR / "intersection_sim.mp4")
LSTM_MODEL_PATH = str(PROJECT_ROOT / "models" / "intersection_sim_trajectories_lstm_model.pt")
TRAJ_DATA_PATH = str(PROJECT_ROOT / "data" / "trajectories" / "intersection_sim_trajectories.json")

WIDTH = 1920
HEIGHT = 1080
FPS = 30
STEP_LENGTH = 1.0 / FPS

# ── Adaptive signal parameters ────────────────────────────────────────────
MIN_GREEN = 10.0   # minimum green time (seconds)
MAX_GREEN = 60.0   # maximum green time (seconds)
YELLOW_DUR = 3.0   # yellow phase duration (seconds)
CYCLE_TOTAL = 90.0 # total cycle budget for green (NS + EW)

# Weight factors for the scoring function
W_QUEUE = 1.0      # weight for current queue count
W_ARRIVAL = 3.0    # weight for arrival rate (veh/s → amplified)
W_THROUGH = 0.3    # bonus weight for through-traffic (clears faster)
ARRIVAL_LOOKAHEAD = 10.0  # seconds to project arrival rate forward

# TLS phase states (20 links)
NS_GREEN  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW = "rrrrryyyyyrrrrryyyyy"

# SUMO edge → direction mapping
EDGE_TO_DIR = {
    "N_to_C": "NB", "S_to_C": "SB",
    "E_to_C": "EB", "W_to_C": "WB",
}
NS_EDGES = ["N_to_C", "S_to_C"]
EW_EDGES = ["E_to_C", "W_to_C"]
NS_DIRS = {"NB", "SB"}
EW_DIRS = {"EB", "WB"}

# Grid cell size matching the trained model (50px cells on 1920x1080 video)
# In SUMO coords: network spans ~200m × ~200m, video is 1920×1080
# We map SUMO positions to pixel coords, then to grid cells
CELL_SIZE = 50  # pixels — same as LSTM training

CAR_COLORS = [
    (200, 30, 30, 255), (30, 70, 180, 255), (230, 230, 230, 255),
    (40, 40, 40, 255), (180, 180, 180, 255), (10, 130, 50, 255),
    (220, 180, 50, 255), (80, 80, 80, 255), (170, 80, 40, 255),
    (100, 50, 150, 255),
]


# ═══════════════════════════════════════════════════════════════════════════
# LSTM Model (same architecture as training)
# ═══════════════════════════════════════════════════════════════════════════

class TurnPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, num_classes))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = self.dropout(hidden[-1])
        return self.fc(out)


class LSTMPredictor:
    """Wraps the trained LSTM for real-time exit prediction."""

    def __init__(self, model_path, device='cpu'):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.label_map = ckpt["label_map"]
        self.inv_map = {v: k for k, v in self.label_map.items()}
        self.clusters = ckpt["clusters"]
        self.cell_size = ckpt["cell_size"]
        self.wait_priors = {}
        if "wait_priors" in ckpt:
            for item in ckpt["wait_priors"]:
                self.wait_priors[tuple(item["cell"])] = item["probs"]

        config = ckpt["model_config"]
        self.model = TurnPredictor(
            input_size=config["input_size"], hidden_size=config["hidden_size"],
            num_layers=config["num_layers"], num_classes=config["num_classes"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.model.to(device)

        # Map exit labels to cardinal directions for signal logic
        # exit clusters near edges → direction
        self.exit_to_direction = {}
        grid_w = WIDTH // self.cell_size
        grid_h = HEIGHT // self.cell_size
        for cl in self.clusters:
            cx, cy = cl["center"]
            label = cl["label"]
            # Determine which edge this exit is near
            if cy <= 2:
                self.exit_to_direction[label] = "NB"  # exits top = northbound
            elif cy >= grid_h - 3:
                self.exit_to_direction[label] = "SB"
            elif cx <= 2:
                self.exit_to_direction[label] = "WB"
            elif cx >= grid_w - 3:
                self.exit_to_direction[label] = "EB"
            else:
                self.exit_to_direction[label] = "unknown"

        print(f"LSTM loaded: {len(self.label_map)} exits")
        for cl in self.clusters:
            d = self.exit_to_direction[cl["label"]]
            print(f"  {cl['label']}: center={cl['center']}, count={cl['count']} → {d}")

    def predict(self, cell_sequence):
        """Predict exit from partial cell sequence → (label, confidence, probs)"""
        if len(cell_sequence) < 3:
            if cell_sequence:
                start = tuple(cell_sequence[0])
                if start in self.wait_priors:
                    probs = self.wait_priors[start]
                    label = max(probs, key=probs.get)
                    return label, probs[label], probs
            return None, 0.0, {}

        features = []
        for i, (x, y) in enumerate(cell_sequence):
            dx = (x - cell_sequence[i-1][0]) if i > 0 else 0.0
            dy = (y - cell_sequence[i-1][1]) if i > 0 else 0.0
            features.append([x / 40.0, y / 22.0, dx / 5.0, dy / 5.0])

        seq = torch.tensor([features], dtype=torch.float32).to(self.device)
        length = torch.tensor([len(cell_sequence)], dtype=torch.long)

        with torch.no_grad():
            output = self.model(seq, length)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        pred_idx = np.argmax(probs)
        label = self.inv_map[pred_idx]
        all_probs = {self.inv_map[i]: float(p) for i, p in enumerate(probs)}
        return label, float(probs[pred_idx]), all_probs

    def is_through_movement(self, entry_dir, exit_label):
        """Check if an entry→exit pair is a through movement (opposite direction)."""
        exit_dir = self.exit_to_direction.get(exit_label, "")
        opposites = {"NB": "SB", "SB": "NB", "EB": "WB", "WB": "EB"}
        return opposites.get(entry_dir) == exit_dir


# ═══════════════════════════════════════════════════════════════════════════
# SUMO → Grid coordinate mapping
# ═══════════════════════════════════════════════════════════════════════════

class SUMOGridMapper:
    """Maps SUMO network coordinates to video-grid cells."""

    def __init__(self, net_bounds, video_w, video_h, cell_size):
        """
        net_bounds: ((x_min, y_min), (x_max, y_max)) from SUMO network
        """
        self.net_xmin, self.net_ymin = net_bounds[0]
        self.net_xmax, self.net_ymax = net_bounds[1]
        self.net_w = self.net_xmax - self.net_xmin
        self.net_h = self.net_ymax - self.net_ymin
        self.video_w = video_w
        self.video_h = video_h
        self.cell_size = cell_size

    def sumo_to_pixel(self, x, y):
        """Convert SUMO (x, y) to pixel (px, py). Note: SUMO y is flipped."""
        px = (x - self.net_xmin) / self.net_w * self.video_w
        py = (1.0 - (y - self.net_ymin) / self.net_h) * self.video_h  # flip y
        return px, py

    def sumo_to_cell(self, x, y):
        """Convert SUMO (x, y) to grid cell (col, row)."""
        px, py = self.sumo_to_pixel(x, y)
        col = int(px // self.cell_size)
        row = int(py // self.cell_size)
        return (col, row)


# ═══════════════════════════════════════════════════════════════════════════
# Signal optimizer — combines queue + arrival rate + LSTM predictions
# ═══════════════════════════════════════════════════════════════════════════

def compute_demand_score(queue_count, arrival_rate, through_ratio):
    """
    Compute a demand score for a direction group (NS or EW).

    Args:
        queue_count:   number of queued vehicles on this group's approaches
        arrival_rate:  combined λ (veh/s) for this group's approaches
        through_ratio: fraction of vehicles predicted to go through (0-1)

    Returns:
        float demand score — higher means more green time needed

    Logic:
        - Queue count is the base demand (vehicles waiting NOW)
        - Arrival rate projects future demand over the lookahead window
        - Through traffic gets a bonus because through movements clear
          faster than turns, so more through = more efficient green usage
          → we actually need LESS time per vehicle but MORE vehicles to clear
    """
    projected_arrivals = arrival_rate * ARRIVAL_LOOKAHEAD
    through_bonus = 1.0 + W_THROUGH * through_ratio  # 1.0 to 1.3

    score = (W_QUEUE * queue_count + W_ARRIVAL * projected_arrivals) * through_bonus
    return score


def compute_green_split(ns_score, ew_score):
    """Split green time proportional to demand scores."""
    total = ns_score + ew_score
    if total < 0.01:
        return CYCLE_TOTAL / 2, CYCLE_TOTAL / 2

    ns_green = CYCLE_TOTAL * (ns_score / total)
    ew_green = CYCLE_TOTAL * (ew_score / total)

    ns_green = max(MIN_GREEN, min(MAX_GREEN, ns_green))
    ew_green = max(MIN_GREEN, min(MAX_GREEN, ew_green))

    return ns_green, ew_green


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    sumo_gui = shutil.which("sumo-gui")
    if not sumo_gui:
        raise FileNotFoundError("sumo-gui not found")
    print(f"Using: {sumo_gui}")

    # ── Load LSTM model ───────────────────────────────────────────────
    print("\n── Loading LSTM model ──")
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"WARNING: LSTM model not found at {LSTM_MODEL_PATH}")
        print("  Running without LSTM predictions (queue + arrival only)")
        lstm = None
    else:
        lstm = LSTMPredictor(LSTM_MODEL_PATH)

    # ── Load trajectory data for QueueManager bootstrap ───────────────
    print("\n── Loading trajectory data for queue manager ──")
    queue_mgr = None
    if os.path.exists(TRAJ_DATA_PATH):
        with open(TRAJ_DATA_PATH) as f:
            traj_data = json.load(f)
        grid_cols = traj_data["grid_cols"]
        grid_w = WIDTH // CELL_SIZE + 1
        grid_h = HEIGHT // CELL_SIZE + 1
        queue_mgr = QueueManager(grid_w, grid_h, fps=FPS)
        queue_mgr.auto_discover_approaches(traj_data["trajectories"], grid_cols)
        queue_mgr.calibrate_with_cell_size(CELL_SIZE)
    else:
        print(f"WARNING: Trajectory data not found at {TRAJ_DATA_PATH}")
        print("  Running without QueueManager")

    # ── Arrival rate estimator ────────────────────────────────────────
    arrival_est = ArrivalRateEstimator(fps=FPS, log_enabled=False)

    # ── Map approach labels to direction groups ───────────────────────
    # QueueManager discovers approaches as A0, A1, ... We need to map
    # these to NS/EW based on their center positions
    approach_to_dir = {}
    if queue_mgr and queue_mgr.approaches:
        grid_w_half = (WIDTH // CELL_SIZE) / 2
        grid_h_half = (HEIGHT // CELL_SIZE) / 2
        for a in queue_mgr.approaches:
            cx, cy = a["center"]
            # Determine if this approach is N/S or E/W by position
            dx = abs(cx - grid_w_half)
            dy = abs(cy - grid_h_half)
            if dy > dx:
                # More vertical displacement → N or S approach
                if cy < grid_h_half:
                    approach_to_dir[a["label"]] = "NB"
                else:
                    approach_to_dir[a["label"]] = "SB"
            else:
                # More horizontal displacement → E or W approach
                if cx < grid_w_half:
                    approach_to_dir[a["label"]] = "WB"
                else:
                    approach_to_dir[a["label"]] = "EB"
        print(f"Approach → direction mapping: {approach_to_dir}")

    # ── Start SUMO ────────────────────────────────────────────────────
    sumo_cmd = [
        sumo_gui, "-c", SUMO_CFG,
        "--start", "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0", "--gui-testing",
        "--step-length", str(STEP_LENGTH),
        "--gui-settings-file", str(SCRIPT_DIR / "intersection.view.xml"),
    ]

    traci.start(sumo_cmd)
    print("\nsumo-gui started.")

    for _ in range(10):
        traci.simulationStep()

    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 100, 100)
    traci.gui.setZoom("View #0", 350)

    for _ in range(5):
        traci.simulationStep()

    # Get network bounds for coordinate mapping
    net_boundary = traci.simulation.getNetBoundary()
    mapper = SUMOGridMapper(net_boundary, WIDTH, HEIGHT, CELL_SIZE)
    print(f"Network bounds: {net_boundary}")

    # Take over TLS control
    tls_id = "C"
    traci.trafficlight.setRedYellowGreenState(tls_id, NS_GREEN)

    screenshot_dir = str(SCRIPT_DIR / "_frames")
    os.makedirs(screenshot_dir, exist_ok=True)

    # ── Tracking state ────────────────────────────────────────────────
    frame_count = 0
    colored_vehicles = set()
    vehicle_cells = defaultdict(list)   # {vid: [(col,row), ...]} trajectory
    vehicle_approach = {}               # {vid: "A0"/"A1"/...}
    vehicle_entry_dir = {}              # {vid: "NB"/"SB"/"EB"/"WB"}
    prev_cells = {}
    prev_vids = set()

    # Signal state machine
    current_phase = "NS_GREEN"
    phase_timer = 0.0
    ns_green_dur = CYCLE_TOTAL / 2
    ew_green_dur = CYCLE_TOTAL / 2
    cycle_count = 0

    print(f"\nAdaptive control: cycle_budget={CYCLE_TOTAL}s, "
          f"min_green={MIN_GREEN}s, max_green={MAX_GREEN}s")
    print(f"Weights: queue={W_QUEUE}, arrival={W_ARRIVAL}, through={W_THROUGH}")
    print(f"LSTM: {'ACTIVE' if lstm else 'DISABLED'}")
    print(f"QueueManager: {'ACTIVE' if queue_mgr else 'DISABLED'}")
    print("Capturing frames...\n")

    frame_paths = []
    decision_log = []

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            phase_timer += STEP_LENGTH
            frame_count += 1
            sim_time = traci.simulation.getTime()

            current_vids = set()
            active_cells = {}

            # ── Track all vehicles ────────────────────────────────────
            for vid in traci.vehicle.getIDList():
                current_vids.add(vid)

                # Random color on spawn
                if vid not in colored_vehicles:
                    traci.vehicle.setColor(vid, random.choice(CAR_COLORS))
                    colored_vehicles.add(vid)

                # Get position and map to grid
                x, y = traci.vehicle.getPosition(vid)
                px, py = mapper.sumo_to_pixel(x, y)
                cell = mapper.sumo_to_cell(x, y)
                active_cells[vid] = cell

                # Determine entry direction from edge
                road_id = traci.vehicle.getRoadID(vid)
                if vid not in vehicle_entry_dir and road_id in EDGE_TO_DIR:
                    vehicle_entry_dir[vid] = EDGE_TO_DIR[road_id]

                # Feed QueueManager
                if queue_mgr:
                    had_approach = vid in queue_mgr.vehicle_approach
                    queue_mgr.track_vehicle(vid, cell, prev_cells.get(vid),
                                            pixel_pos=(px, py))
                    if not had_approach and vid in queue_mgr.vehicle_approach:
                        vehicle_approach[vid] = queue_mgr.vehicle_approach[vid]
                        arrival_est.record_arrival(
                            vid, queue_mgr.vehicle_approach[vid], frame_count)

                # Record trajectory for LSTM
                if vid not in prev_cells or prev_cells[vid] != cell:
                    vehicle_cells[vid].append(cell)
                prev_cells[vid] = cell

            # ── Handle disappeared vehicles ───────────────────────────
            for vid in prev_vids - current_vids:
                if queue_mgr:
                    queue_mgr.vehicle_exited(vid)
                arrival_est.vehicle_exited(vid)
            prev_vids = current_vids

            # ── Phase transitions ─────────────────────────────────────
            phase_switch = False

            if current_phase == "NS_GREEN" and phase_timer >= ns_green_dur:
                traci.trafficlight.setRedYellowGreenState(tls_id, NS_YELLOW)
                current_phase = "NS_YELLOW"
                phase_timer = 0.0

            elif current_phase == "NS_YELLOW" and phase_timer >= YELLOW_DUR:
                # ── DECISION POINT: compute next cycle's green split ──
                ns_score, ew_score, details = _compute_scores(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, sim_time)
                ns_green_dur, ew_green_dur = compute_green_split(ns_score, ew_score)

                traci.trafficlight.setRedYellowGreenState(tls_id, EW_GREEN)
                current_phase = "EW_GREEN"
                phase_timer = 0.0
                cycle_count += 1

                print(f"  Cycle {cycle_count} @ t={sim_time:.1f}s: "
                      f"NS_score={ns_score:.1f} EW_score={ew_score:.1f} → "
                      f"NS_green={ns_green_dur:.1f}s EW_green={ew_green_dur:.1f}s")
                print(f"    {details}")
                decision_log.append({
                    "cycle": cycle_count, "time": sim_time,
                    "ns_score": ns_score, "ew_score": ew_score,
                    "ns_green": ns_green_dur, "ew_green": ew_green_dur,
                    "details": details})

            elif current_phase == "EW_GREEN" and phase_timer >= ew_green_dur:
                traci.trafficlight.setRedYellowGreenState(tls_id, EW_YELLOW)
                current_phase = "EW_YELLOW"
                phase_timer = 0.0

            elif current_phase == "EW_YELLOW" and phase_timer >= YELLOW_DUR:
                ns_score, ew_score, details = _compute_scores(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, sim_time)
                ns_green_dur, ew_green_dur = compute_green_split(ns_score, ew_score)

                traci.trafficlight.setRedYellowGreenState(tls_id, NS_GREEN)
                current_phase = "NS_GREEN"
                phase_timer = 0.0
                cycle_count += 1

                print(f"  Cycle {cycle_count} @ t={sim_time:.1f}s: "
                      f"NS_score={ns_score:.1f} EW_score={ew_score:.1f} → "
                      f"NS_green={ns_green_dur:.1f}s EW_green={ew_green_dur:.1f}s")
                print(f"    {details}")
                decision_log.append({
                    "cycle": cycle_count, "time": sim_time,
                    "ns_score": ns_score, "ew_score": ew_score,
                    "ns_green": ns_green_dur, "ew_green": ew_green_dur,
                    "details": details})

            # ── Capture frame ─────────────────────────────────────────
            fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
            traci.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
            frame_paths.append(fpath)

            if frame_count % 300 == 0:
                n = traci.vehicle.getIDCount()
                print(f"  Frame {frame_count}, sim time: {sim_time:.1f}s, vehicles: {n}")

    except traci.exceptions.FatalTraCIError:
        print("Simulation ended.")
    finally:
        try:
            traci.simulationStep()
        except Exception:
            pass
        traci.close()

    # ── Encode video ──────────────────────────────────────────────────
    print(f"Captured {frame_count} frames. Encoding video...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
    written = 0

    for fpath in frame_paths:
        for _ in range(10):
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                break
            time.sleep(0.01)
        if os.path.exists(fpath):
            frame = cv2.imread(fpath)
            if frame is not None:
                if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                    frame = cv2.resize(frame, (WIDTH, HEIGHT))
                writer.write(frame)
                written += 1

    writer.release()

    for fpath in frame_paths:
        if os.path.exists(fpath):
            os.remove(fpath)
    try:
        os.rmdir(screenshot_dir)
    except OSError:
        pass

    duration = written / FPS
    print(f"\nVideo: {OUTPUT_VIDEO}")
    print(f"Frames: {written}, Duration: {duration:.1f}s at {FPS} FPS")

    # ── Save decision log ─────────────────────────────────────────────
    log_path = str(SCRIPT_DIR / "decision_log.json")
    with open(log_path, "w") as f:
        json.dump(decision_log, f, indent=2)
    print(f"Decision log: {log_path}")


def _compute_scores(queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, sim_time):
    """
    Compute demand scores for NS and EW using all available signals.

    Returns: (ns_score, ew_score, details_string)
    """
    # 1. Queue counts per direction group
    ns_queue = 0
    ew_queue = 0

    if queue_mgr:
        queue_state = queue_mgr.compute_state(active_cells)
        for approach_label, data in queue_state.items():
            direction = approach_to_dir.get(approach_label)
            if direction in NS_DIRS:
                ns_queue += data["queue"]
            elif direction in EW_DIRS:
                ew_queue += data["queue"]
    else:
        # Fallback: use TraCI halting count
        for edge in NS_EDGES:
            ns_queue += traci.edge.getLastStepHaltingNumber(edge)
        for edge in EW_EDGES:
            ew_queue += traci.edge.getLastStepHaltingNumber(edge)

    # 2. Arrival rates per direction group
    ns_arrival = 0.0
    ew_arrival = 0.0

    all_rates = arrival_est.compute_all_rates(frame_count)
    for approach_label, rate in all_rates.items():
        direction = approach_to_dir.get(approach_label)
        if direction in NS_DIRS:
            ns_arrival += rate
        elif direction in EW_DIRS:
            ew_arrival += rate

    # 3. LSTM turn predictions — compute through-ratio per group
    ns_through_ratio = 0.5  # default: assume 50% through
    ew_through_ratio = 0.5

    if lstm:
        ns_through = 0
        ns_total = 0
        ew_through = 0
        ew_total = 0

        for vid, cells in vehicle_cells.items():
            if vid not in active_cells:
                continue  # skip exited vehicles
            entry_dir = vehicle_entry_dir.get(vid)
            if not entry_dir:
                continue

            pred_label, confidence, _ = lstm.predict(cells)
            if pred_label is None or confidence < 0.4:
                continue

            is_through = lstm.is_through_movement(entry_dir, pred_label)

            if entry_dir in NS_DIRS:
                ns_total += 1
                if is_through:
                    ns_through += 1
            elif entry_dir in EW_DIRS:
                ew_total += 1
                if is_through:
                    ew_through += 1

        if ns_total > 0:
            ns_through_ratio = ns_through / ns_total
        if ew_total > 0:
            ew_through_ratio = ew_through / ew_total

    # 4. Compute combined scores
    ns_score = compute_demand_score(ns_queue, ns_arrival, ns_through_ratio)
    ew_score = compute_demand_score(ew_queue, ew_arrival, ew_through_ratio)

    details = (f"queues: NS={ns_queue} EW={ew_queue} | "
               f"arrivals: NS={ns_arrival:.2f} EW={ew_arrival:.2f} veh/s | "
               f"through: NS={ns_through_ratio:.0%} EW={ew_through_ratio:.0%}")

    return ns_score, ew_score, details


if __name__ == "__main__":
    main()
