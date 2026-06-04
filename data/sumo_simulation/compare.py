"""
Compare fixed vs adaptive timer on identical traffic.
Runs headless (no GUI) and prints key metrics.

Usage:
    python compare.py           # defaults to v1
    python compare.py v2        # run v2 scenario

Adaptive controller uses the full pipeline:
  - QueueManager (velocity-based queue detection)
  - ArrivalRateEstimator (rolling-window λ per approach)
  - LSTM TurnPredictor (exit intention from partial trajectories)
"""

import os
import sys
import json
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path

import traci
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Parse version from command line
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"

from queue_manager import QueueManager
from arrival_rate import ArrivalRateEstimator

STEP_LENGTH = 0.1  # headless: 10 steps/sec for accuracy
FPS = 1.0 / STEP_LENGTH  # virtual FPS for arrival rate estimator

# Paths
LSTM_MODEL_PATH = str(PROJECT_ROOT / "models" / "intersection_sim_trajectories_lstm_model.pt")
TRAJ_DATA_PATH = str(PROJECT_ROOT / "data" / "trajectories" / "intersection_sim_trajectories.json")

# Adaptive parameters (same as run_sumo_video.py)
MIN_GREEN = 10.0
MAX_GREEN = 60.0
YELLOW_DUR = 3.0
CYCLE_TOTAL = 90.0
W_QUEUE = 1.0
W_ARRIVAL = 3.0
W_THROUGH = 0.3
ARRIVAL_LOOKAHEAD = 10.0

NS_GREEN  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW = "rrrrryyyyyrrrrryyyyy"

EDGE_TO_DIR = {"N_to_C": "NB", "S_to_C": "SB", "E_to_C": "EB", "W_to_C": "WB"}
NS_EDGES = ["N_to_C", "S_to_C"]
EW_EDGES = ["E_to_C", "W_to_C"]
NS_DIRS = {"NB", "SB"}
EW_DIRS = {"EB", "WB"}

# Video dimensions for grid mapping (same as training)
WIDTH, HEIGHT, CELL_SIZE = 1920, 1080, 50


# ── LSTM model ────────────────────────────────────────────────────────────

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
        return self.fc(self.dropout(hidden[-1]))


class LSTMPredictor:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.label_map = ckpt["label_map"]
        self.inv_map = {v: k for k, v in self.label_map.items()}
        self.clusters = ckpt["clusters"]
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

        grid_w = WIDTH // CELL_SIZE
        grid_h = HEIGHT // CELL_SIZE
        self.exit_to_direction = {}
        for cl in self.clusters:
            cx, cy = cl["center"]
            label = cl["label"]
            if cy <= 2: self.exit_to_direction[label] = "NB"
            elif cy >= grid_h - 3: self.exit_to_direction[label] = "SB"
            elif cx <= 2: self.exit_to_direction[label] = "WB"
            elif cx >= grid_w - 3: self.exit_to_direction[label] = "EB"
            else: self.exit_to_direction[label] = "unknown"

    def predict(self, cell_sequence):
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

        seq = torch.tensor([features], dtype=torch.float32)
        length = torch.tensor([len(cell_sequence)], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(seq, length), dim=1).squeeze().cpu().numpy()
        idx = np.argmax(probs)
        return self.inv_map[idx], float(probs[idx]), {self.inv_map[i]: float(p) for i, p in enumerate(probs)}

    def is_through(self, entry_dir, exit_label):
        exit_dir = self.exit_to_direction.get(exit_label, "")
        opposites = {"NB": "SB", "SB": "NB", "EB": "WB", "WB": "EB"}
        return opposites.get(entry_dir) == exit_dir


# ── Grid mapper ───────────────────────────────────────────────────────────

class GridMapper:
    def __init__(self, net_bounds):
        self.xmin, self.ymin = net_bounds[0]
        self.xmax, self.ymax = net_bounds[1]
        self.w = self.xmax - self.xmin
        self.h = self.ymax - self.ymin

    def to_pixel(self, x, y):
        px = (x - self.xmin) / self.w * WIDTH
        py = (1.0 - (y - self.ymin) / self.h) * HEIGHT
        return px, py

    def to_cell(self, x, y):
        px, py = self.to_pixel(x, y)
        return (int(px // CELL_SIZE), int(py // CELL_SIZE))


# ── Score computation ─────────────────────────────────────────────────────

def compute_demand_score(queue_count, arrival_rate, through_ratio):
    projected = arrival_rate * ARRIVAL_LOOKAHEAD
    bonus = 1.0 + W_THROUGH * through_ratio
    return (W_QUEUE * queue_count + W_ARRIVAL * projected) * bonus


def compute_green_split(ns_score, ew_score):
    total = ns_score + ew_score
    if total < 0.01:
        return CYCLE_TOTAL / 2, CYCLE_TOTAL / 2
    ns = max(MIN_GREEN, min(MAX_GREEN, CYCLE_TOTAL * ns_score / total))
    ew = max(MIN_GREEN, min(MAX_GREEN, CYCLE_TOTAL * ew_score / total))
    return ns, ew


def compute_scores(queue_mgr, arrival_est, lstm, vehicle_cells,
                   vehicle_entry_dir, approach_to_dir, active_cells, frame_count):
    ns_queue = ew_queue = 0
    if queue_mgr:
        state = queue_mgr.compute_state(active_cells)
        for label, data in state.items():
            d = approach_to_dir.get(label)
            if d in NS_DIRS: ns_queue += data["queue"]
            elif d in EW_DIRS: ew_queue += data["queue"]
    else:
        for e in NS_EDGES: ns_queue += traci.edge.getLastStepHaltingNumber(e)
        for e in EW_EDGES: ew_queue += traci.edge.getLastStepHaltingNumber(e)

    ns_arrival = ew_arrival = 0.0
    for label, rate in arrival_est.compute_all_rates(frame_count).items():
        d = approach_to_dir.get(label)
        if d in NS_DIRS: ns_arrival += rate
        elif d in EW_DIRS: ew_arrival += rate

    ns_tr, ew_tr = 0.5, 0.5
    if lstm:
        ns_thr, ns_tot, ew_thr, ew_tot = 0, 0, 0, 0
        for vid, cells in vehicle_cells.items():
            if vid not in active_cells: continue
            entry = vehicle_entry_dir.get(vid)
            if not entry: continue
            pred, conf, _ = lstm.predict(cells)
            if pred is None or conf < 0.4: continue
            through = lstm.is_through(entry, pred)
            if entry in NS_DIRS:
                ns_tot += 1
                if through: ns_thr += 1
            elif entry in EW_DIRS:
                ew_tot += 1
                if through: ew_thr += 1
        if ns_tot > 0: ns_tr = ns_thr / ns_tot
        if ew_tot > 0: ew_tr = ew_thr / ew_tot

    return (compute_demand_score(ns_queue, ns_arrival, ns_tr),
            compute_demand_score(ew_queue, ew_arrival, ew_tr))


# ── Run fixed timer ──────────────────────────────────────────────────────

def run_fixed():
    sumo = shutil.which("sumo")
    cfg = str(SCRIPT_DIR / f"fixed_timer/{VERSION}/intersection.sumocfg")
    traci.start([sumo, "-c", cfg, "--no-step-log", "--step-length", str(STEP_LENGTH)])

    total_waiting = 0.0
    departed = arrived = max_veh = 0
    # Track per-vehicle last-known waiting time
    veh_last_wait = {}  # {vid: last accumulated waiting time}
    veh_total_wait = []  # final waiting time per vehicle at exit

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        n = traci.vehicle.getIDCount()
        total_waiting += n
        if n > max_veh: max_veh = n
        departed += traci.simulation.getDepartedNumber()
        arrived += traci.simulation.getArrivedNumber()

        # Track accumulated waiting time for all active vehicles
        for vid in traci.vehicle.getIDList():
            veh_last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

        # Record final waiting time for exiting vehicles
        for vid in traci.simulation.getArrivedIDList():
            if vid in veh_last_wait:
                veh_total_wait.append(veh_last_wait.pop(vid))

    sim_time = traci.simulation.getTime()
    traci.close()

    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    return {
        "sim_time": sim_time, "departed": departed, "arrived": arrived,
        "max_vehicles": max_veh,
        "total_vehicle_seconds": total_waiting * STEP_LENGTH,
        "total_wait_time": total_wait,
    }


# ── Run adaptive timer (full pipeline) ───────────────────────────────────

def run_adaptive():
    sumo = shutil.which("sumo")
    cfg = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection.sumocfg")
    traci.start([sumo, "-c", cfg, "--no-step-log", "--step-length", str(STEP_LENGTH)])

    traci.simulationStep()

    # Get network bounds
    net_bounds = traci.simulation.getNetBoundary()
    mapper = GridMapper(net_bounds)

    # Load LSTM
    lstm = LSTMPredictor(LSTM_MODEL_PATH) if os.path.exists(LSTM_MODEL_PATH) else None

    # Load QueueManager
    queue_mgr = None
    approach_to_dir = {}
    if os.path.exists(TRAJ_DATA_PATH):
        with open(TRAJ_DATA_PATH) as f:
            td = json.load(f)
        grid_w = WIDTH // CELL_SIZE + 1
        grid_h = HEIGHT // CELL_SIZE + 1
        queue_mgr = QueueManager(grid_w, grid_h, fps=FPS)
        queue_mgr.auto_discover_approaches(td["trajectories"], td["grid_cols"])
        queue_mgr.calibrate_with_cell_size(CELL_SIZE)

        gw_half = (WIDTH // CELL_SIZE) / 2
        gh_half = (HEIGHT // CELL_SIZE) / 2
        for a in queue_mgr.approaches:
            cx, cy = a["center"]
            dx, dy = abs(cx - gw_half), abs(cy - gh_half)
            if dy > dx:
                approach_to_dir[a["label"]] = "NB" if cy < gh_half else "SB"
            else:
                approach_to_dir[a["label"]] = "WB" if cx < gw_half else "EB"

    arrival_est = ArrivalRateEstimator(fps=FPS, log_enabled=False)

    # Take over TLS
    traci.trafficlight.setRedYellowGreenState("C", NS_GREEN)

    current_phase = "NS_GREEN"
    phase_timer = 0.0
    ns_green_dur = ew_green_dur = CYCLE_TOTAL / 2

    vehicle_cells = defaultdict(list)
    vehicle_entry_dir = {}
    prev_cells = {}
    prev_vids = set()
    frame_count = 0

    total_waiting = 0.0
    departed = arrived = max_veh = 0
    veh_last_wait = {}
    veh_total_wait = []

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        phase_timer += STEP_LENGTH
        frame_count += 1

        current_vids = set()
        active_cells = {}

        for vid in traci.vehicle.getIDList():
            current_vids.add(vid)
            x, y = traci.vehicle.getPosition(vid)
            px, py = mapper.to_pixel(x, y)
            cell = mapper.to_cell(x, y)
            active_cells[vid] = cell

            road = traci.vehicle.getRoadID(vid)
            if vid not in vehicle_entry_dir and road in EDGE_TO_DIR:
                vehicle_entry_dir[vid] = EDGE_TO_DIR[road]

            if queue_mgr:
                had = vid in queue_mgr.vehicle_approach
                queue_mgr.track_vehicle(vid, cell, prev_cells.get(vid), pixel_pos=(px, py))
                if not had and vid in queue_mgr.vehicle_approach:
                    arrival_est.record_arrival(vid, queue_mgr.vehicle_approach[vid], frame_count)

            if vid not in prev_cells or prev_cells[vid] != cell:
                vehicle_cells[vid].append(cell)
            prev_cells[vid] = cell

            veh_last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

        # Record final waiting time for exiting vehicles
        for vid in traci.simulation.getArrivedIDList():
            if vid in veh_last_wait:
                veh_total_wait.append(veh_last_wait.pop(vid))

        for vid in prev_vids - current_vids:
            if queue_mgr: queue_mgr.vehicle_exited(vid)
            arrival_est.vehicle_exited(vid)
        prev_vids = current_vids

        n = traci.vehicle.getIDCount()
        total_waiting += n
        if n > max_veh: max_veh = n
        departed += traci.simulation.getDepartedNumber()
        arrived += traci.simulation.getArrivedNumber()

        # Phase transitions
        if current_phase == "NS_GREEN" and phase_timer >= ns_green_dur:
            traci.trafficlight.setRedYellowGreenState("C", NS_YELLOW)
            current_phase = "NS_YELLOW"
            phase_timer = 0.0
        elif current_phase == "NS_YELLOW" and phase_timer >= YELLOW_DUR:
            ns_s, ew_s = compute_scores(queue_mgr, arrival_est, lstm, vehicle_cells,
                                         vehicle_entry_dir, approach_to_dir, active_cells, frame_count)
            ns_green_dur, ew_green_dur = compute_green_split(ns_s, ew_s)
            traci.trafficlight.setRedYellowGreenState("C", EW_GREEN)
            current_phase = "EW_GREEN"
            phase_timer = 0.0
        elif current_phase == "EW_GREEN" and phase_timer >= ew_green_dur:
            traci.trafficlight.setRedYellowGreenState("C", EW_YELLOW)
            current_phase = "EW_YELLOW"
            phase_timer = 0.0
        elif current_phase == "EW_YELLOW" and phase_timer >= YELLOW_DUR:
            ns_s, ew_s = compute_scores(queue_mgr, arrival_est, lstm, vehicle_cells,
                                         vehicle_entry_dir, approach_to_dir, active_cells, frame_count)
            ns_green_dur, ew_green_dur = compute_green_split(ns_s, ew_s)
            traci.trafficlight.setRedYellowGreenState("C", NS_GREEN)
            current_phase = "NS_GREEN"
            phase_timer = 0.0

    sim_time = traci.simulation.getTime()
    traci.close()

    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    return {
        "sim_time": sim_time, "departed": departed, "arrived": arrived,
        "max_vehicles": max_veh,
        "total_vehicle_seconds": total_waiting * STEP_LENGTH,
        "total_wait_time": total_wait,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"FIXED TIMER vs ADAPTIVE TIMER (Full Pipeline) — {VERSION} Comparison")
    print("=" * 65)

    print("\nRunning fixed timer...")
    fixed = run_fixed()

    print("\nRunning adaptive timer (LSTM + Queue + Arrival Rate)...")
    adaptive = run_adaptive()

    print("\n" + "-" * 65)
    print(f"{'Metric':<40} {'Fixed':>10} {'Adaptive':>10}")
    print("-" * 65)

    print(f"{'Simulation end time (s)':<40} {fixed['sim_time']:>10.1f} {adaptive['sim_time']:>10.1f}")
    print(f"{'Total vehicles departed':<40} {fixed['departed']:>10} {adaptive['departed']:>10}")
    print(f"{'Total vehicles arrived':<40} {fixed['arrived']:>10} {adaptive['arrived']:>10}")
    print(f"{'Peak concurrent vehicles':<40} {fixed['max_vehicles']:>10} {adaptive['max_vehicles']:>10}")
    print(f"{'Total vehicle-seconds in network':<40} {fixed['total_vehicle_seconds']:>10.1f} {adaptive['total_vehicle_seconds']:>10.1f}")
    print(f"{'Total waiting time (veh-seconds)':<40} {fixed['total_wait_time']:>10.1f} {adaptive['total_wait_time']:>10.1f}")

    fixed_avg = fixed['total_vehicle_seconds'] / max(fixed['arrived'], 1)
    adaptive_avg = adaptive['total_vehicle_seconds'] / max(adaptive['arrived'], 1)
    fixed_wait = fixed['total_wait_time'] / max(fixed['arrived'], 1)
    adaptive_wait = adaptive['total_wait_time'] / max(adaptive['arrived'], 1)

    print(f"{'Avg time per vehicle (s)':<40} {fixed_avg:>10.2f} {adaptive_avg:>10.2f}")
    print(f"{'Avg waiting time per vehicle (s)':<40} {fixed_wait:>10.2f} {adaptive_wait:>10.2f}")

    # Determine winner on avg time
    diff_time = ((adaptive_avg - fixed_avg) / fixed_avg) * 100 if fixed_avg > 0 else 0
    winner_time = "ADAPTIVE" if adaptive_avg < fixed_avg else "FIXED"

    diff_wait = ((adaptive_wait - fixed_wait) / fixed_wait) * 100 if fixed_wait > 0 else 0
    winner_wait = "ADAPTIVE" if adaptive_wait < fixed_wait else "FIXED"

    print(f"\n>> Avg time in network: {winner_time} wins by {abs(diff_time):.1f}%")
    print(f">> Avg waiting time:    {winner_wait} wins by {abs(diff_wait):.1f}%")
    print("-" * 65)


if __name__ == "__main__":
    main()
