(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: data/sumo_simulation/adaptive_algorithm.md
================================================
# Adaptive Signal Control Algorithm

## Overview

The adaptive controller uses three data sources to allocate green time:

1. **QueueManager** — velocity-based queue detection per approach (from `queue_manager.py`)
2. **ArrivalRateEstimator** — rolling-window λ per approach in veh/s (from `arrival_rate.py`)
3. **LSTM TurnPredictor** — predicts exit direction from partial trajectory (from `lstm_predictor.py`)

At every phase switch, these three signals are combined into a **demand score** per direction group (NS vs EW), and green time is split proportionally.

---

## Signal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_GREEN` | 10 s | Minimum green time for any phase |
| `MAX_GREEN` | 60 s | Maximum green time for any phase |
| `YELLOW_DUR` | 3 s | Yellow transition between phases |
| `CYCLE_TOTAL` | 90 s | Total green budget shared between NS and EW |

## Scoring Weights

| Weight | Value | Description |
|--------|-------|-------------|
| `W_QUEUE` | 1.0 | Weight for current queue count (vehicles stopped) |
| `W_ARRIVAL` | 3.0 | Weight for arrival rate signal |
| `W_THROUGH` | 0.3 | Bonus multiplier for through-traffic ratio |
| `ARRIVAL_LOOKAHEAD` | 10 s | Seconds to project arrival rate forward |

## QueueManager Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VELOCITY_THRESHOLD` | 1.0 m/s | Below this speed, vehicle is "queued" |
| `SMOOTHING_FRAMES` | 5 frames | Window for velocity averaging |
| `APPROACH_RADIUS` | 8 cells | Clustering radius for approach discovery |
| `MIN_APPROACH_COUNT` | 21 | Minimum trajectories to form an approach |
| `INTERSECTION_MARGIN` | 0.35 | Center 35% of grid excluded as intersection |

## ArrivalRateEstimator Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SECONDS` | 15 s | Rolling window for λ computation |

## LSTM Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 4 | Features per step: (x, y, dx, dy) |
| `hidden_size` | 64 | LSTM hidden units |
| `num_layers` | 2 | Stacked LSTM layers |
| `num_classes` | 4 | Exit zones (auto-discovered) |
| `dropout` | 0.3 | Dropout rate |
| `min_steps` | 3 | Minimum trajectory length before LSTM predicts |
| `confidence_threshold` | 0.4 | Below this, LSTM prediction is ignored |
| `cell_size` | 50 px | Grid cell size for trajectory discretization |

---

## Phase Cycle

```
NS_GREEN → NS_YELLOW (3s) → EW_GREEN → EW_YELLOW (3s) → repeat
```

The order is fixed. Only the **duration** of each green phase changes cycle to cycle.

---

## Decision Algorithm

At every phase transition (yellow → green), the controller:

### Step 1: Measure current queues
```
ns_queue = queued vehicles on N_to_C + S_to_C approaches
ew_queue = queued vehicles on E_to_C + W_to_C approaches
```
Uses QueueManager's velocity-based detection (speed < 1.0 m/s = stopped).

### Step 2: Measure arrival rates
```
ns_arrival = λ(North approach) + λ(South approach)   [veh/s]
ew_arrival = λ(East approach) + λ(West approach)     [veh/s]
```
Rolling 15-second window from ArrivalRateEstimator.

### Step 3: Predict through-traffic ratio via LSTM
For each active vehicle with ≥ 3 trajectory steps:
- Feed partial cell sequence to LSTM
- Get predicted exit zone + confidence
- If confidence ≥ 0.4, classify as through or turning
```
ns_through_ratio = (# NS vehicles predicted through) / (# NS vehicles predicted)
ew_through_ratio = (# EW vehicles predicted through) / (# EW vehicles predicted)
```
Default: 0.5 if no predictions available.

### Step 4: Compute demand scores
```
demand_score = (W_QUEUE × queue + W_ARRIVAL × arrival_rate × LOOKAHEAD) × (1 + W_THROUGH × through_ratio)
```

Expanded:
```
ns_score = (1.0 × ns_queue + 3.0 × ns_arrival × 10.0) × (1.0 + 0.3 × ns_through_ratio)
ew_score = (1.0 × ew_queue + 3.0 × ew_arrival × 10.0) × (1.0 + 0.3 × ew_through_ratio)
```

### Step 5: Split green time
```
ns_green = 90 × (ns_score / (ns_score + ew_score))
ew_green = 90 × (ew_score / (ns_score + ew_score))
```
Both clamped to [10, 60] seconds.

If both scores are 0: equal split (45s / 45s).

---

## Why each component matters

| Component | What it captures | Without it |
|-----------|-----------------|------------|
| **Queue count** | Immediate demand — vehicles stopped right now | Ignores current congestion |
| **Arrival rate** | Demand trend — is traffic building or subsiding? | Reacts too late to surges |
| **LSTM through-ratio** | Movement efficiency — through traffic clears faster than turns | Overallocates green to turn-heavy approaches |

---

## Experimental Results (seed=42)

| Scenario | Type | Avg Time (Fixed) | Avg Time (Adaptive) | Improvement |
|----------|------|------------------:|--------------------:|------------:|
| v1 | Balanced baseline | 56.65 s | 56.58 s | +0.1% |
| v2 | Demand flip | 60.48 s | 56.54 s | +6.5% |
| v3 | Morning CBD rush (NYC) | 64.51 s | 56.51 s | +12.4% |
| v4 | Evening exodus (LA) | 60.04 s | 55.87 s | +6.9% |
| v5 | School zone pulse | 44.59 s | 50.75 s | -13.8% |
| v6 | Oversaturated (Mumbai) | 63.12 s | 66.35 s | -5.1% |
| v7 | Stadium dispersal | 45.29 s | 37.71 s | +16.7% |

Adaptive wins 4/7 scenarios. Best on asymmetric/time-varying demand (v3, v7).
Fixed wins on short spikes (v5) and fully saturated conditions (v6).



================================================
FILE: data/sumo_simulation/build_net.sh
================================================
#!/bin/bash
# Build the SUMO network from node + edge files
# Run this once, or whenever you change .nod.xml / .edg.xml

netconvert \
    --node-files=intersection.nod.xml \
    --edge-files=intersection.edg.xml \
    --output-file=intersection.net.xml \
    --tls.guess true

echo "Network built: intersection.net.xml"
echo "Run with:  sumo-gui -c intersection.sumocfg"



================================================
FILE: data/sumo_simulation/compare.py
================================================
"""
Compare fixed vs adaptive timer on identical traffic.
Runs with sumo-gui, captures videos for both, and prints detailed info
at every signal change.

Usage:
    python compare.py           # defaults to v2
    python compare.py v2        # run v2 scenario

Adaptive controller uses the full pipeline:
  - QueueManager (velocity-based queue detection)
  - ArrivalRateEstimator (rolling-window λ per approach)
  - LSTM TurnPredictor (exit intention from partial trajectories)
"""

import os
import sys
import time
import json
import shutil
import random
from collections import defaultdict
from pathlib import Path

import traci
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Parse version from command line
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"

from queue_manager import QueueManager
from arrival_rate import ArrivalRateEstimator
from lstm_predictor import RealtimePredictor

STEP_LENGTH = 1.0 / 30  # match video FPS
FPS = 30

# Paths — auto-resolve per version
LSTM_MODEL_PATH = str(PROJECT_ROOT / "models" / f"fixed_{VERSION}_trajectories_lstm_model.pt")
TRAJ_DATA_PATH = str(PROJECT_ROOT / "data" / "transitions" / "sumo_transition" / f"fixed_{VERSION}_trajectories.json")

# Adaptive parameters
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

WIDTH, HEIGHT, CELL_SIZE = 1920, 1080, 50

CAR_COLORS = [
    (200, 30, 30, 255), (30, 70, 180, 255), (230, 230, 230, 255),
    (40, 40, 40, 255), (180, 180, 180, 255), (10, 130, 50, 255),
    (220, 180, 50, 255), (80, 80, 80, 255), (170, 80, 40, 255),
    (100, 50, 150, 255),
]


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


def compute_max_pressure_split(queue_mgr, approach_to_dir, active_cells):
    """
    Max-pressure controller (Varaiya 2013).
    Give green to whichever phase has more queued vehicles.
    No arrival rate, no LSTM — pure queue comparison.
    """
    ns_queue = ew_queue = 0

    if queue_mgr:
        state = queue_mgr.compute_state(active_cells)
        for label, data in state.items():
            d = approach_to_dir.get(label)
            if d in NS_DIRS:   ns_queue += data["queue"]
            elif d in EW_DIRS: ew_queue += data["queue"]
    else:
        for e in NS_EDGES: ns_queue += traci.edge.getLastStepHaltingNumber(e)
        for e in EW_EDGES: ew_queue += traci.edge.getLastStepHaltingNumber(e)

    if ns_queue >= ew_queue:
        return MAX_GREEN, MIN_GREEN
    else:
        return MIN_GREEN, MAX_GREEN


def compute_scores_verbose(queue_mgr, arrival_est, lstm, vehicle_cells,
                           vehicle_entry_dir, approach_to_dir, active_cells,
                           frame_count, sim_time, cycle_count, next_phase):
    """Compute scores and print every detail."""
    # 1. Queue counts
    ns_queue = ew_queue = 0
    queue_detail = {}
    if queue_mgr:
        state = queue_mgr.compute_state(active_cells)
        for label, data in state.items():
            d = approach_to_dir.get(label)
            queue_detail[label] = {"dir": d, "queued": data["queue"], "total": data["total"]}
            if d in NS_DIRS: ns_queue += data["queue"]
            elif d in EW_DIRS: ew_queue += data["queue"]
    else:
        for e in NS_EDGES: ns_queue += traci.edge.getLastStepHaltingNumber(e)
        for e in EW_EDGES: ew_queue += traci.edge.getLastStepHaltingNumber(e)

    # 2. Arrival rates
    ns_arrival = ew_arrival = 0.0
    arrival_detail = {}
    all_rates = arrival_est.compute_all_rates(frame_count)
    for label, rate in all_rates.items():
        d = approach_to_dir.get(label)
        arrival_detail[label] = {"dir": d, "rate": rate}
        if d in NS_DIRS: ns_arrival += rate
        elif d in EW_DIRS: ew_arrival += rate

    # 3. LSTM predictions per vehicle
    ns_tr, ew_tr = 0.5, 0.5
    ns_thr, ns_tot, ew_thr, ew_tot = 0, 0, 0, 0
    lstm_predictions = []
    if lstm:
        for vid, cells in vehicle_cells.items():
            if vid not in active_cells:
                continue
            entry = vehicle_entry_dir.get(vid)
            if not entry:
                continue
            pred, conf, all_probs = lstm.predict(cells)
            if pred is None or conf < 0.4:
                continue
            through = lstm.is_through_movement(entry, pred)
            lstm_predictions.append({
                "vid": vid, "entry": entry, "pred": pred,
                "conf": conf, "through": through, "steps": len(cells)
            })
            if entry in NS_DIRS:
                ns_tot += 1
                if through: ns_thr += 1
            elif entry in EW_DIRS:
                ew_tot += 1
                if through: ew_thr += 1
        if ns_tot > 0: ns_tr = ns_thr / ns_tot
        if ew_tot > 0: ew_tr = ew_thr / ew_tot

    # 4. Compute scores
    ns_score = compute_demand_score(ns_queue, ns_arrival, ns_tr)
    ew_score = compute_demand_score(ew_queue, ew_arrival, ew_tr)
    ns_green, ew_green = compute_green_split(ns_score, ew_score)

    # 5. Print everything
    n_vehicles = len(active_cells)
    total_wait_now = sum(traci.vehicle.getAccumulatedWaitingTime(v) for v in active_cells)
    avg_wait_now = total_wait_now / n_vehicles if n_vehicles > 0 else 0

    print(f"\n{'='*75}")
    print(f"  SIGNAL CHANGE — Cycle {cycle_count} @ t={sim_time:.1f}s")
    print(f"  Switching to: {next_phase}")
    print(f"{'='*75}")

    print(f"\n  VEHICLES IN NETWORK: {n_vehicles}")
    print(f"  TOTAL WAIT RIGHT NOW: {total_wait_now:.1f}s (avg {avg_wait_now:.1f}s/vehicle)")

    # Per-edge vehicle counts + speeds
    print(f"\n  ── Per-Edge Status ──")
    for edge in NS_EDGES + EW_EDGES:
        count = traci.edge.getLastStepVehicleNumber(edge)
        halting = traci.edge.getLastStepHaltingNumber(edge)
        speed = traci.edge.getLastStepMeanSpeed(edge)
        direction = EDGE_TO_DIR[edge]
        print(f"    {edge} ({direction}): {count} vehicles, {halting} halting, "
              f"avg speed {speed:.1f} m/s")

    # Queue details
    print(f"\n  ── Queue Detection (QueueManager) ──")
    if queue_detail:
        for label, d in sorted(queue_detail.items()):
            print(f"    {label} ({d['dir']}): {d['queued']} queued / {d['total']} total")
    print(f"    TOTAL → NS: {ns_queue} queued | EW: {ew_queue} queued")

    # Arrival rates
    print(f"\n  ── Arrival Rates ──")
    for label, d in sorted(arrival_detail.items()):
        print(f"    {label} ({d['dir']}): {d['rate']:.3f} veh/s")
    print(f"    TOTAL → NS: {ns_arrival:.3f} veh/s | EW: {ew_arrival:.3f} veh/s")
    print(f"    Projected arrivals (next {ARRIVAL_LOOKAHEAD}s): "
          f"NS: {ns_arrival * ARRIVAL_LOOKAHEAD:.1f} | EW: {ew_arrival * ARRIVAL_LOOKAHEAD:.1f}")

    # LSTM predictions
    print(f"\n  ── LSTM Predictions ──")
    if lstm_predictions:
        for p in sorted(lstm_predictions, key=lambda x: x["vid"]):
            mov = "THROUGH" if p["through"] else "TURN"
            print(f"    Vehicle {p['vid']}: entry={p['entry']} → pred={p['pred']} "
                  f"({p['conf']:.0%} conf, {p['steps']} steps) [{mov}]")
        print(f"    Through ratios → NS: {ns_thr}/{ns_tot} ({ns_tr:.0%}) | "
              f"EW: {ew_thr}/{ew_tot} ({ew_tr:.0%})")
    else:
        print(f"    No confident predictions this cycle")

    # Final scoring
    print(f"\n  ── Scoring ──")
    print(f"    NS: queue={ns_queue} × {W_QUEUE} + arrival={ns_arrival:.2f} × {W_ARRIVAL} × "
          f"lookahead={ARRIVAL_LOOKAHEAD} → through_bonus={1 + W_THROUGH * ns_tr:.2f}")
    print(f"    EW: queue={ew_queue} × {W_QUEUE} + arrival={ew_arrival:.2f} × {W_ARRIVAL} × "
          f"lookahead={ARRIVAL_LOOKAHEAD} → through_bonus={1 + W_THROUGH * ew_tr:.2f}")
    print(f"    NS_SCORE = {ns_score:.2f} | EW_SCORE = {ew_score:.2f}")

    # Green allocation
    print(f"\n  ── Green Allocation ──")
    ns_pct = ns_score / (ns_score + ew_score) * 100 if (ns_score + ew_score) > 0.01 else 50
    ew_pct = 100 - ns_pct
    print(f"    NS: {ns_green:.1f}s ({ns_pct:.0f}%) | EW: {ew_green:.1f}s ({ew_pct:.0f}%)")
    print(f"    (min={MIN_GREEN}s, max={MAX_GREEN}s, budget={CYCLE_TOTAL}s)")
    print(f"{'='*75}")

    return ns_score, ew_score, ns_green, ew_green


# ── Encode video from frames ─────────────────────────────────────────────

def save_metrics(result, mode, version):
    """Save all metrics to a JSON file in the version's directory."""
    arrived = max(result['arrived'], 1)
    metrics = {
        "version": version,
        "mode": mode,
        "simulation_duration_s": round(result['sim_time'], 1),
        "video_frames": result['frames'],
        "total_vehicles_departed": result['departed'],
        "total_vehicles_arrived": result['arrived'],
        "peak_concurrent_vehicles": result['max_vehicles'],
        "total_vehicle_seconds": round(result['total_vehicle_seconds'], 1),
        "total_waiting_time_veh_s": round(result['total_wait_time'], 1),
        "avg_time_per_vehicle_s": round(result['total_vehicle_seconds'] / arrived, 2),
        "avg_waiting_time_per_vehicle_s": round(result['total_wait_time'] / arrived, 2),
    }

    if mode == "fixed":
        out_dir = SCRIPT_DIR / f"fixed_timer/{version}"
    else:
        out_dir = SCRIPT_DIR / f"adaptive_timer/{version}"

    filename = f"metrics_{mode}.json"
    out_path = out_dir / filename
    os.makedirs(str(out_dir), exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {out_path}")
    return metrics


def encode_video(frame_paths, output_path, frame_count):
    print(f"\nEncoding {frame_count} frames → {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
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
    print(f"  Written: {written} frames, {written/FPS:.1f}s")
    return written


# ── Run fixed timer (GUI + video) ────────────────────────────────────────

def run_fixed():
    sumo_gui = shutil.which("sumo-gui")
    cfg = str(SCRIPT_DIR / f"fixed_timer/{VERSION}/intersection.sumocfg")
    view_xml = str(SCRIPT_DIR / f"fixed_timer/{VERSION}/intersection.view.xml")
    output_video = str(SCRIPT_DIR / f"fixed_timer/{VERSION}/intersection_sim.mp4")

    sumo_cmd = [
        sumo_gui, "-c", cfg,
        "--start", "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0", "--gui-testing",
        "--step-length", str(STEP_LENGTH),
    ]
    if os.path.exists(view_xml):
        sumo_cmd += ["--gui-settings-file", view_xml]

    traci.start(sumo_cmd)
    print("\n[FIXED] sumo-gui started.")

    for _ in range(10):
        traci.simulationStep()
    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 100, 100)
    traci.gui.setZoom("View #0", 350)
    for _ in range(5):
        traci.simulationStep()

    screenshot_dir = str(SCRIPT_DIR / f"fixed_timer/{VERSION}/_frames")
    os.makedirs(screenshot_dir, exist_ok=True)

    frame_count = 0
    colored = set()
    total_waiting = 0.0
    departed = arrived = max_veh = 0
    veh_last_wait = {}
    veh_total_wait = []
    frame_paths = []

    print("[FIXED] Running with default signal timing...")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            frame_count += 1
            sim_time = traci.simulation.getTime()

            for vid in traci.vehicle.getIDList():
                if vid not in colored:
                    traci.vehicle.setColor(vid, random.choice(CAR_COLORS))
                    colored.add(vid)
                veh_last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

            for vid in traci.simulation.getArrivedIDList():
                if vid in veh_last_wait:
                    veh_total_wait.append(veh_last_wait.pop(vid))

            n = traci.vehicle.getIDCount()
            total_waiting += n
            if n > max_veh: max_veh = n
            departed += traci.simulation.getDepartedNumber()
            arrived += traci.simulation.getArrivedNumber()

            fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
            traci.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
            frame_paths.append(fpath)

            if frame_count % 300 == 0:
                print(f"  [FIXED] Frame {frame_count}, t={sim_time:.1f}s, vehicles={n}")

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try: traci.simulationStep()
        except: pass
        traci.close()

    written = encode_video(frame_paths, output_video, frame_count)
    try: os.rmdir(screenshot_dir)
    except: pass

    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    result = {
        "sim_time": frame_count * STEP_LENGTH,
        "departed": departed, "arrived": arrived,
        "max_vehicles": max_veh,
        "total_vehicle_seconds": total_waiting * STEP_LENGTH,
        "total_wait_time": total_wait,
        "frames": written,
    }
    save_metrics(result, "fixed", VERSION)
    return result


# ── Run adaptive timer (GUI + video + verbose) ──────────────────────────

def run_adaptive():
    sumo_gui = shutil.which("sumo-gui")
    cfg = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection.sumocfg")
    view_xml = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection.view.xml")
    output_video = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection_sim.mp4")

    sumo_cmd = [
        sumo_gui, "-c", cfg,
        "--start", "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0", "--gui-testing",
        "--step-length", str(STEP_LENGTH),
    ]
    if os.path.exists(view_xml):
        sumo_cmd += ["--gui-settings-file", view_xml]

    traci.start(sumo_cmd)
    print("\n[ADAPTIVE] sumo-gui started.")

    for _ in range(10):
        traci.simulationStep()
    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 100, 100)
    traci.gui.setZoom("View #0", 350)
    for _ in range(5):
        traci.simulationStep()

    net_bounds = traci.simulation.getNetBoundary()
    mapper = GridMapper(net_bounds)

    # Load LSTM
    lstm = RealtimePredictor(LSTM_MODEL_PATH) if os.path.exists(LSTM_MODEL_PATH) else None
    print(f"[ADAPTIVE] LSTM: {'ACTIVE' if lstm else 'DISABLED'}")

    # Load QueueManager
    queue_mgr = None
    approach_to_dir = {}
    if os.path.exists(TRAJ_DATA_PATH):
        with open(TRAJ_DATA_PATH) as f:
            td = json.load(f)
        grid_w = WIDTH // CELL_SIZE + 1
        grid_h = HEIGHT // CELL_SIZE + 1
        queue_mgr = QueueManager(grid_w, grid_h, fps=FPS, intersection_dist=INTERSECTION_DIST, intersection_margin=INTERSECTION_MARGIN)
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
        print(f"[ADAPTIVE] Approach mapping: {approach_to_dir}")
    print(f"[ADAPTIVE] QueueManager: {'ACTIVE' if queue_mgr else 'DISABLED'}")

    arrival_est = ArrivalRateEstimator(fps=FPS, log_enabled=False)

    # Take over TLS
    traci.trafficlight.setRedYellowGreenState("C", NS_GREEN)

    screenshot_dir = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/_frames")
    os.makedirs(screenshot_dir, exist_ok=True)

    current_phase = "NS_GREEN"
    phase_timer = 0.0
    ns_green_dur = ew_green_dur = CYCLE_TOTAL / 2
    cycle_count = 0

    vehicle_cells = defaultdict(list)
    vehicle_entry_dir = {}
    prev_cells = {}
    prev_vids = set()
    frame_count = 0
    colored = set()

    total_waiting = 0.0
    departed = arrived = max_veh = 0
    veh_last_wait = {}
    veh_total_wait = []
    frame_paths = []
    decision_log = []

    print(f"[ADAPTIVE] Weights: queue={W_QUEUE}, arrival={W_ARRIVAL}, through={W_THROUGH}")
    print(f"[ADAPTIVE] Cycle budget={CYCLE_TOTAL}s, min={MIN_GREEN}s, max={MAX_GREEN}s")
    print("[ADAPTIVE] Running...\n")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            phase_timer += STEP_LENGTH
            frame_count += 1
            sim_time = traci.simulation.getTime()

            current_vids = set()
            active_cells = {}

            for vid in traci.vehicle.getIDList():
                current_vids.add(vid)
                if vid not in colored:
                    traci.vehicle.setColor(vid, random.choice(CAR_COLORS))
                    colored.add(vid)

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

            # Phase transitions with verbose output
            if current_phase == "NS_GREEN" and phase_timer >= ns_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", NS_YELLOW)
                current_phase = "NS_YELLOW"
                phase_timer = 0.0

            elif current_phase == "NS_YELLOW" and phase_timer >= YELLOW_DUR:
                cycle_count += 1
                ns_s, ew_s, ns_green_dur, ew_green_dur = compute_scores_verbose(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, sim_time, cycle_count, "EW_GREEN")
                decision_log.append({
                    "cycle": cycle_count, "time": round(sim_time, 1),
                    "ns_score": round(ns_s, 2), "ew_score": round(ew_s, 2),
                    "ns_green": round(ns_green_dur, 1), "ew_green": round(ew_green_dur, 1),
                    "vehicles": n,
                })
                traci.trafficlight.setRedYellowGreenState("C", EW_GREEN)
                current_phase = "EW_GREEN"
                phase_timer = 0.0

            elif current_phase == "EW_GREEN" and phase_timer >= ew_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", EW_YELLOW)
                current_phase = "EW_YELLOW"
                phase_timer = 0.0

            elif current_phase == "EW_YELLOW" and phase_timer >= YELLOW_DUR:
                cycle_count += 1
                ns_s, ew_s, ns_green_dur, ew_green_dur = compute_scores_verbose(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, sim_time, cycle_count, "NS_GREEN")
                decision_log.append({
                    "cycle": cycle_count, "time": round(sim_time, 1),
                    "ns_score": round(ns_s, 2), "ew_score": round(ew_s, 2),
                    "ns_green": round(ns_green_dur, 1), "ew_green": round(ew_green_dur, 1),
                    "vehicles": n,
                })
                traci.trafficlight.setRedYellowGreenState("C", NS_GREEN)
                current_phase = "NS_GREEN"
                phase_timer = 0.0

            # Capture frame
            fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
            traci.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
            frame_paths.append(fpath)

            if frame_count % 300 == 0:
                print(f"  [ADAPTIVE] Frame {frame_count}, t={sim_time:.1f}s, vehicles={n}")

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try: traci.simulationStep()
        except: pass
        traci.close()

    written = encode_video(frame_paths, output_video, frame_count)
    try: os.rmdir(screenshot_dir)
    except: pass

    # Save decision log
    log_path = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/decision_log.json")
    with open(log_path, "w") as f:
        json.dump(decision_log, f, indent=2)
    print(f"  Decision log: {log_path}")

    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    result = {
        "sim_time": frame_count * STEP_LENGTH,
        "departed": departed, "arrived": arrived,
        "max_vehicles": max_veh,
        "total_vehicle_seconds": total_waiting * STEP_LENGTH,
        "total_wait_time": total_wait,
        "frames": written,
    }
    save_metrics(result, "adaptive", VERSION)
    return result


# ── Run max-pressure timer ───────────────────────────────────────────────

def run_max_pressure():
    sumo_gui = shutil.which("sumo-gui")
    cfg = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection.sumocfg")
    view_xml = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection.view.xml")
    output_video = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/intersection_maxpressure.mp4")

    sumo_cmd = [
        sumo_gui, "-c", cfg,
        "--start", "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0", "--gui-testing",
        "--step-length", str(STEP_LENGTH),
    ]
    if os.path.exists(view_xml):
        sumo_cmd += ["--gui-settings-file", view_xml]

    traci.start(sumo_cmd)
    print("\n[MAX-PRESSURE] sumo-gui started.")

    for _ in range(10): traci.simulationStep()
    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 100, 100)
    traci.gui.setZoom("View #0", 350)
    for _ in range(5): traci.simulationStep()

    net_bounds = traci.simulation.getNetBoundary()
    mapper = GridMapper(net_bounds)

    queue_mgr = None
    approach_to_dir = {}
    if os.path.exists(TRAJ_DATA_PATH):
        with open(TRAJ_DATA_PATH) as f:
            td = json.load(f)
        grid_w = WIDTH // CELL_SIZE + 1
        grid_h = HEIGHT // CELL_SIZE + 1
        queue_mgr = QueueManager(grid_w, grid_h, fps=FPS, intersection_dist=INTERSECTION_DIST, intersection_margin=INTERSECTION_MARGIN)
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
        print(f"[MAX-PRESSURE] Approach mapping: {approach_to_dir}")
    print(f"[MAX-PRESSURE] QueueManager: {'ACTIVE' if queue_mgr else 'DISABLED'}")

    traci.trafficlight.setRedYellowGreenState("C", NS_GREEN)

    screenshot_dir = str(SCRIPT_DIR / f"adaptive_timer/{VERSION}/_frames_mp")
    os.makedirs(screenshot_dir, exist_ok=True)

    current_phase = "NS_GREEN"
    phase_timer = 0.0
    ns_green_dur = ew_green_dur = CYCLE_TOTAL / 2

    prev_cells = {}
    prev_vids = set()
    frame_count = 0
    colored = set()
    total_waiting = 0.0
    departed = arrived = max_veh = 0
    veh_last_wait = {}
    veh_total_wait = []
    frame_paths = []

    print("[MAX-PRESSURE] Running...\n")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            phase_timer += STEP_LENGTH
            frame_count += 1
            sim_time = traci.simulation.getTime()

            current_vids = set()
            active_cells = {}

            for vid in traci.vehicle.getIDList():
                current_vids.add(vid)
                if vid not in colored:
                    traci.vehicle.setColor(vid, random.choice(CAR_COLORS))
                    colored.add(vid)

                x, y = traci.vehicle.getPosition(vid)
                px, py = mapper.to_pixel(x, y)
                cell = mapper.to_cell(x, y)
                active_cells[vid] = cell

                if queue_mgr:
                    queue_mgr.track_vehicle(vid, cell, prev_cells.get(vid), pixel_pos=(px, py))

                prev_cells[vid] = cell
                veh_last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

            for vid in traci.simulation.getArrivedIDList():
                if vid in veh_last_wait:
                    veh_total_wait.append(veh_last_wait.pop(vid))

            for vid in prev_vids - current_vids:
                if queue_mgr: queue_mgr.vehicle_exited(vid)
            prev_vids = current_vids

            n = traci.vehicle.getIDCount()
            total_waiting += n
            if n > max_veh: max_veh = n
            departed += traci.simulation.getDepartedNumber()
            arrived += traci.simulation.getArrivedNumber()

            if current_phase == "NS_GREEN" and phase_timer >= ns_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", NS_YELLOW)
                current_phase = "NS_YELLOW"
                phase_timer = 0.0

            elif current_phase == "NS_YELLOW" and phase_timer >= YELLOW_DUR:
                ns_green_dur, ew_green_dur = compute_max_pressure_split(
                    queue_mgr, approach_to_dir, active_cells)
                traci.trafficlight.setRedYellowGreenState("C", EW_GREEN)
                current_phase = "EW_GREEN"
                phase_timer = 0.0

            elif current_phase == "EW_GREEN" and phase_timer >= ew_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", EW_YELLOW)
                current_phase = "EW_YELLOW"
                phase_timer = 0.0

            elif current_phase == "EW_YELLOW" and phase_timer >= YELLOW_DUR:
                ns_green_dur, ew_green_dur = compute_max_pressure_split(
                    queue_mgr, approach_to_dir, active_cells)
                traci.trafficlight.setRedYellowGreenState("C", NS_GREEN)
                current_phase = "NS_GREEN"
                phase_timer = 0.0

            fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
            traci.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
            frame_paths.append(fpath)

            if frame_count % 300 == 0:
                print(f"  [MAX-PRESSURE] Frame {frame_count}, t={sim_time:.1f}s, vehicles={n}")

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try: traci.simulationStep()
        except: pass
        traci.close()

    written = encode_video(frame_paths, output_video, frame_count)
    try: os.rmdir(screenshot_dir)
    except: pass

    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    result = {
        "sim_time": frame_count * STEP_LENGTH,
        "departed": departed, "arrived": arrived,
        "max_vehicles": max_veh,
        "total_vehicle_seconds": total_waiting * STEP_LENGTH,
        "total_wait_time": total_wait,
        "frames": written,
    }
    save_metrics(result, "maxpressure", VERSION)
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print(f"  FIXED vs MAX-PRESSURE vs ADAPTIVE — {VERSION} (GUI + Video + Verbose)")
    print("=" * 75)

    print("\n" + "#" * 75)
    print("  PHASE 1: FIXED TIMER")
    print("#" * 75)
    fixed = run_fixed()

    print("\n" + "#" * 75)
    print("  PHASE 2: MAX-PRESSURE")
    print("#" * 75)
    mp = run_max_pressure()

    print("\n" + "#" * 75)
    print("  PHASE 3: ADAPTIVE TIMER (LSTM + Queue + Arrival)")
    print("#" * 75)
    adaptive = run_adaptive()

    # ── Final comparison ──────────────────────────────────────────────
    fixed_avg    = fixed['total_vehicle_seconds']    / max(fixed['arrived'], 1)
    mp_avg       = mp['total_vehicle_seconds']        / max(mp['arrived'], 1)
    adaptive_avg = adaptive['total_vehicle_seconds'] / max(adaptive['arrived'], 1)

    fixed_wait    = fixed['total_wait_time']    / max(fixed['arrived'], 1)
    mp_wait       = mp['total_wait_time']        / max(mp['arrived'], 1)
    adaptive_wait = adaptive['total_wait_time'] / max(adaptive['arrived'], 1)

    print(f"\n\n{'='*75}")
    print(f"  FINAL COMPARISON — {VERSION}")
    print(f"{'='*75}")
    print(f"\n{'Metric':<45} {'Fixed':>10} {'Max-P':>10} {'Adaptive':>10}")
    print("-" * 75)
    print(f"{'Simulation duration (s)':<45} {fixed['sim_time']:>10.1f} {mp['sim_time']:>10.1f} {adaptive['sim_time']:>10.1f}")
    print(f"{'Video frames captured':<45} {fixed['frames']:>10} {mp['frames']:>10} {adaptive['frames']:>10}")
    print(f"{'Total vehicles departed':<45} {fixed['departed']:>10} {mp['departed']:>10} {adaptive['departed']:>10}")
    print(f"{'Total vehicles arrived':<45} {fixed['arrived']:>10} {mp['arrived']:>10} {adaptive['arrived']:>10}")
    print(f"{'Peak concurrent vehicles':<45} {fixed['max_vehicles']:>10} {mp['max_vehicles']:>10} {adaptive['max_vehicles']:>10}")
    print(f"{'Total vehicle-seconds in network':<45} {fixed['total_vehicle_seconds']:>10.1f} {mp['total_vehicle_seconds']:>10.1f} {adaptive['total_vehicle_seconds']:>10.1f}")
    print(f"{'Total waiting time (veh-seconds)':<45} {fixed['total_wait_time']:>10.1f} {mp['total_wait_time']:>10.1f} {adaptive['total_wait_time']:>10.1f}")
    print(f"{'Avg time per vehicle (s)':<45} {fixed_avg:>10.2f} {mp_avg:>10.2f} {adaptive_avg:>10.2f}")
    print(f"{'Avg waiting time per vehicle (s)':<45} {fixed_wait:>10.2f} {mp_wait:>10.2f} {adaptive_wait:>10.2f}")
    print("-" * 75)

    def pct(new, base): return ((new - base) / base * 100) if base > 0 else 0

    print(f"\n  vs Fixed:   Max-P {pct(mp_wait, fixed_wait):+.1f}%  |  Adaptive {pct(adaptive_wait, fixed_wait):+.1f}%")
    print(f"  vs Max-P:                      Adaptive {pct(adaptive_wait, mp_wait):+.1f}%")

    print(f"\n  Videos saved:")
    print(f"    Fixed:        fixed_timer/{VERSION}/intersection_sim.mp4")
    print(f"    Max-Pressure: adaptive_timer/{VERSION}/intersection_maxpressure.mp4")
    print(f"    Adaptive:     adaptive_timer/{VERSION}/intersection_sim.mp4")
    print(f"{'='*75}")


if __name__ == "__main__":
    MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
    # Optional: intersection distance in meters (e.g., python compare.py v3 all 200)
    INTERSECTION_DIST = float(sys.argv[3]) if len(sys.argv) > 3 else None
    # Optional: intersection margin 0.0-0.5 (e.g., python compare.py v3 all 200 0.25)
    INTERSECTION_MARGIN = float(sys.argv[4]) if len(sys.argv) > 4 else None

    if MODE == "maxpressure":
        result = run_max_pressure()
        avg_time = result['total_vehicle_seconds'] / max(result['arrived'], 1)
        avg_wait = result['total_wait_time'] / max(result['arrived'], 1)
        print(f"\n{'='*75}")
        print(f"  MAX-PRESSURE RESULTS — {VERSION}")
        print(f"{'='*75}")
        print(f"\n{'Metric':<45} {'Value':>12}")
        print("-" * 60)
        print(f"{'Simulation duration (s)':<45} {result['sim_time']:>12.1f}")
        print(f"{'Video frames captured':<45} {result['frames']:>12}")
        print(f"{'Total vehicles departed':<45} {result['departed']:>12}")
        print(f"{'Total vehicles arrived':<45} {result['arrived']:>12}")
        print(f"{'Peak concurrent vehicles':<45} {result['max_vehicles']:>12}")
        print(f"{'Total vehicle-seconds in network':<45} {result['total_vehicle_seconds']:>12.1f}")
        print(f"{'Total waiting time (veh-seconds)':<45} {result['total_wait_time']:>12.1f}")
        print(f"{'Avg time per vehicle (s)':<45} {avg_time:>12.2f}")
        print(f"{'Avg waiting time per vehicle (s)':<45} {avg_wait:>12.2f}")
        print("-" * 60)
    elif MODE == "adaptive":
        run_adaptive()
    elif MODE == "fixed":
        run_fixed()
    else:
        main()



================================================
FILE: data/sumo_simulation/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-05-30T17:39:12.764222-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

    <tls_building>
        <tls.guess value="true"/>
    </tls_building>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/intersection.rou.xml
================================================
<routes>
    <!-- Vehicle types with proper SUMO GUI shapes -->
    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.70"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="car2"  length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.00"
           guiShape="passenger" color="0.1,0.3,0.7"/>
    <vType id="car3"  length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.00"
           guiShape="passenger" color="0.9,0.9,0.9"/>
    <vType id="car4"  length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.00"
           guiShape="passenger" color="0.2,0.2,0.2"/>
    <vType id="car5"  length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.00"
           guiShape="passenger" color="0.0,0.5,0.2"/>
    <vType id="van"   length="5.75" width="2.2" minGap="2.5" maxSpeed="7.50"  accel="1.8" decel="4.0" sigma="0.5" probability="0.15"
           guiShape="delivery" color="0.9,0.8,0.3"/>
    <vType id="truck" length="10.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.10"
           guiShape="truck" color="0.6,0.4,0.2"/>
    <vType id="bus"   length="12.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.05"
           guiShape="bus" color="0.2,0.6,0.8"/>

    <!-- Routes -->
    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>

    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>

    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>

    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- Flows with random vehicle color assignment -->
    <!-- North: 0.3 veh/s -->
    <flow id="N_f_through" route="N_through" begin="0" end="120" probability="0.18" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_f_left"    route="N_left"    begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_f_right"   route="N_right"   begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.3 veh/s -->
    <flow id="S_f_through" route="S_through" begin="0" end="120" probability="0.18" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_f_left"    route="S_left"    begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_f_right"   route="S_right"   begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.2 veh/s -->
    <flow id="E_f_through" route="E_through" begin="0" end="120" probability="0.12" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_f_left"    route="E_left"    begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_f_right"   route="E_right"   begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.2 veh/s -->
    <flow id="W_f_through" route="W_through" begin="0" end="120" probability="0.12" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_f_left"    route="W_left"    begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_f_right"   route="W_right"   begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="120"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/run_sumo_video.py
================================================
"""
SUMO GUI Video Capture
======================
Runs sumo-gui and captures frames via TraCI screenshots.
Uses SUMO's native rendering — proper vehicle shapes, traffic lights, roads.

Usage:
    python run_sumo_video.py
"""

import os
import time
import shutil
import random
import traci
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = os.path.join(SCRIPT_DIR, "intersection.sumocfg")
OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "intersection_sim.mp4")

WIDTH = 1920
HEIGHT = 1080
FPS = 30
STEP_LENGTH = 1.0 / FPS  # one simulation step = one video frame

# Car colors for variety — assigned randomly via TraCI at spawn
CAR_COLORS = [
    (200, 30, 30, 255),    # red
    (30, 70, 180, 255),    # blue
    (230, 230, 230, 255),  # white
    (40, 40, 40, 255),     # black
    (180, 180, 180, 255),  # silver
    (10, 130, 50, 255),    # green
    (220, 180, 50, 255),   # gold
    (80, 80, 80, 255),     # dark gray
    (170, 80, 40, 255),    # brown
    (100, 50, 150, 255),   # purple
]


def main():
    sumo_gui = shutil.which("sumo-gui")
    if not sumo_gui:
        raise FileNotFoundError("sumo-gui not found")
    print(f"Using: {sumo_gui}")

    sumo_cmd = [
        sumo_gui,
        "-c", SUMO_CFG,
        "--start",
        "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0",
        "--gui-testing",
        "--step-length", str(STEP_LENGTH),
        "--gui-settings-file", os.path.join(SCRIPT_DIR, "intersection.view.xml"),
    ]

    traci.start(sumo_cmd)
    print("sumo-gui started.")

    # Wait for GUI to fully initialize
    for _ in range(10):
        traci.simulationStep()

    # Set view — network center is at (100,100)
    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 100, 100)
    traci.gui.setZoom("View #0", 350)

    # Let view settle
    for _ in range(5):
        traci.simulationStep()

    screenshot_dir = os.path.join(SCRIPT_DIR, "_frames")
    os.makedirs(screenshot_dir, exist_ok=True)

    frame_count = 0
    colored_vehicles = set()

    print("Capturing frames...")

    # Phase 1: Run simulation and request all screenshots
    frame_paths = []
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Assign random colors to new vehicles
            for vid in traci.vehicle.getIDList():
                if vid not in colored_vehicles:
                    color = random.choice(CAR_COLORS)
                    traci.vehicle.setColor(vid, color)
                    colored_vehicles.add(vid)

            # Request screenshot — SUMO writes it before the NEXT step completes
            fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
            traci.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
            frame_paths.append(fpath)
            frame_count += 1

            if frame_count % 300 == 0:
                n = traci.vehicle.getIDCount()
                sim_time = traci.simulation.getTime()
                print(f"  Frame {frame_count}, sim time: {sim_time:.1f}s, vehicles: {n}")

    except traci.exceptions.FatalTraCIError:
        print("Simulation ended.")
    finally:
        # One more step to flush the last screenshot
        try:
            traci.simulationStep()
        except Exception:
            pass
        traci.close()

    print(f"Captured {frame_count} frames. Encoding video...")

    # Phase 2: Encode all frames to video at constant FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
    written = 0

    for fpath in frame_paths:
        # Wait briefly if file not yet written
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

    # Cleanup frame images
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


if __name__ == "__main__":
    main()



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T18:09:22.278726-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v1: Baseline traffic demand                         -->
    <!-- N/S: 0.3 veh/s,  E/W: 0.2 veh/s                            -->
    <!-- Turn split: 60% through, 20% left, 20% right                -->
    <!-- Duration: 120s                                               -->
    <!-- Fixed seed for reproducibility                               -->
    <!-- ============================================================ -->

    <!-- Vehicle types -->
    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.70"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="van"   length="5.75" width="2.2" minGap="2.5" maxSpeed="7.50"  accel="1.8" decel="4.0" sigma="0.5" probability="0.15"
           guiShape="delivery" color="0.9,0.8,0.3"/>
    <vType id="truck" length="10.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.10"
           guiShape="truck" color="0.6,0.4,0.2"/>
    <vType id="bus"   length="12.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.05"
           guiShape="bus" color="0.2,0.6,0.8"/>

    <!-- Routes -->
    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- North: 0.3 veh/s -->
    <flow id="N_f_through" route="N_through" begin="0" end="120" probability="0.18" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_f_left"    route="N_left"    begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_f_right"   route="N_right"   begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.3 veh/s -->
    <flow id="S_f_through" route="S_through" begin="0" end="120" probability="0.18" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_f_left"    route="S_left"    begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_f_right"   route="S_right"   begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.2 veh/s -->
    <flow id="E_f_through" route="E_through" begin="0" end="120" probability="0.12" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_f_left"    route="E_left"    begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_f_right"   route="E_right"   begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.2 veh/s -->
    <flow id="W_f_through" route="W_through" begin="0" end="120" probability="0.12" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_f_left"    route="W_left"    begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_f_right"   route="W_right"   begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="120"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/adaptive_timer/v1/run_sumo_video.py
================================================
[Binary file]


================================================
FILE: data/sumo_simulation/adaptive_timer/v2/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v2/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T18:45:43.313058-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v2/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v2/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v2: Unbalanced + time-varying demand                -->
    <!-- Phase 1 (0-60s):  N/S heavy 0.5 veh/s, E/W light 0.1 veh/s -->
    <!-- Phase 2 (60-120s): N/S drops to 0.1, E/W surges to 0.5     -->
    <!-- Turn split: 60% through, 20% left, 20% right                -->
    <!-- Fixed seed for reproducibility                               -->
    <!-- ============================================================ -->

    <!-- Vehicle types -->
    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.70"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="van"   length="5.75" width="2.2" minGap="2.5" maxSpeed="7.50"  accel="1.8" decel="4.0" sigma="0.5" probability="0.15"
           guiShape="delivery" color="0.9,0.8,0.3"/>
    <vType id="truck" length="10.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.10"
           guiShape="truck" color="0.6,0.4,0.2"/>
    <vType id="bus"   length="12.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.05"
           guiShape="bus" color="0.2,0.6,0.8"/>

    <!-- Routes -->
    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- ═══════════════════════════════════════════════════════════ -->
    <!-- PHASE 1: 0-60s — N/S HEAVY (0.5 veh/s), E/W LIGHT (0.1)  -->
    <!-- ═══════════════════════════════════════════════════════════ -->

    <!-- North: 0.5 veh/s -->
    <flow id="N1_through" route="N_through" begin="0" end="60" probability="0.30" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_left"    route="N_left"    begin="0" end="60" probability="0.10" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_right"   route="N_right"   begin="0" end="60" probability="0.10" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.5 veh/s -->
    <flow id="S1_through" route="S_through" begin="0" end="60" probability="0.30" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_left"    route="S_left"    begin="0" end="60" probability="0.10" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_right"   route="S_right"   begin="0" end="60" probability="0.10" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.1 veh/s -->
    <flow id="E1_through" route="E_through" begin="0" end="60" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_left"    route="E_left"    begin="0" end="60" probability="0.02" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_right"   route="E_right"   begin="0" end="60" probability="0.02" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.1 veh/s -->
    <flow id="W1_through" route="W_through" begin="0" end="60" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_left"    route="W_left"    begin="0" end="60" probability="0.02" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_right"   route="W_right"   begin="0" end="60" probability="0.02" departLane="best" departSpeed="desired" type="car"/>

    <!-- ═══════════════════════════════════════════════════════════ -->
    <!-- PHASE 2: 60-120s — DEMAND FLIPS                           -->
    <!-- N/S drops to 0.1, E/W surges to 0.5                      -->
    <!-- ═══════════════════════════════════════════════════════════ -->

    <!-- North: 0.1 veh/s -->
    <flow id="N2_through" route="N_through" begin="60" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_left"    route="N_left"    begin="60" end="120" probability="0.02" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_right"   route="N_right"   begin="60" end="120" probability="0.02" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.1 veh/s -->
    <flow id="S2_through" route="S_through" begin="60" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_left"    route="S_left"    begin="60" end="120" probability="0.02" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_right"   route="S_right"   begin="60" end="120" probability="0.02" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.5 veh/s -->
    <flow id="E2_through" route="E_through" begin="60" end="120" probability="0.30" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_left"    route="E_left"    begin="60" end="120" probability="0.10" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_right"   route="E_right"   begin="60" end="120" probability="0.10" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.5 veh/s -->
    <flow id="W2_through" route="W_through" begin="60" end="120" probability="0.30" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_left"    route="W_left"    begin="60" end="120" probability="0.10" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_right"   route="W_right"   begin="60" end="120" probability="0.10" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v2/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="120"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v2/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/adaptive_timer/v2/run_sumo_video.py
================================================
[Binary file]


================================================
FILE: data/sumo_simulation/adaptive_timer/v3/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v3/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T19:12:31.912486-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v3/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v3/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v3: Morning CBD Rush (NYC/Manhattan-style)          -->
    <!-- Heavy N/S commuter flow, light E/W cross-streets             -->
    <!-- Based on: Manhattan morning peak directional split (65-70%   -->
    <!--   inbound) with typical 70% through, 15% left, 15% right    -->
    <!-- N/S: 0.6 veh/s (heavy commuter corridor)                    -->
    <!-- E/W: 0.15 veh/s (minor cross-streets)                       -->
    <!-- Duration: 180s steady state                                  -->
    <!-- Source: NYC DOT Traffic Volume Counts, HCM PHF=0.92          -->
    <!-- ============================================================ -->

    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33" accel="2.0" decel="4.5" sigma="0.5"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="van"   length="5.75" width="2.2" minGap="2.5" maxSpeed="7.50" accel="1.8" decel="4.0" sigma="0.5"
           guiShape="delivery" color="0.9,0.8,0.3"/>
    <vType id="truck" length="10.0" width="2.5" minGap="3.0" maxSpeed="6.94" accel="1.2" decel="3.5" sigma="0.5"
           guiShape="truck" color="0.6,0.4,0.2"/>

    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- North: 0.6 veh/s (70% through, 15% left, 15% right) -->
    <flow id="N_through" route="N_through" begin="0" end="180" probability="0.42" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_left"    route="N_left"    begin="0" end="180" probability="0.09" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_right"   route="N_right"   begin="0" end="180" probability="0.09" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.6 veh/s -->
    <flow id="S_through" route="S_through" begin="0" end="180" probability="0.42" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_left"    route="S_left"    begin="0" end="180" probability="0.09" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_right"   route="S_right"   begin="0" end="180" probability="0.09" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.15 veh/s (minor cross-street) -->
    <flow id="E_through" route="E_through" begin="0" end="180" probability="0.105" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_left"    route="E_left"    begin="0" end="180" probability="0.023" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_right"   route="E_right"   begin="0" end="180" probability="0.023" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.15 veh/s -->
    <flow id="W_through" route="W_through" begin="0" end="180" probability="0.105" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_left"    route="W_left"    begin="0" end="180" probability="0.023" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_right"   route="W_right"   begin="0" end="180" probability="0.023" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v3/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="180"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v3/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/adaptive_timer/v4/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v4/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T19:12:32.076837-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v4/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v4/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v4: Evening Exodus with Mid-Shift (LA-style)        -->
    <!-- Models downtown LA evening: heavy outbound that gradually    -->
    <!-- shifts direction as commuters clear and nightlife begins     -->
    <!-- Phase 1 (0-90s): Heavy S→N outbound, light others           -->
    <!-- Phase 2 (90-180s): Demand equalizes as rush subsides         -->
    <!-- Based on: LADOT evening peak data, PHF=0.88                  -->
    <!-- ============================================================ -->

    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33" accel="2.0" decel="4.5" sigma="0.5"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="van"   length="5.75" width="2.2" minGap="2.5" maxSpeed="7.50" accel="1.8" decel="4.0" sigma="0.5"
           guiShape="delivery" color="0.9,0.8,0.3"/>
    <vType id="truck" length="10.0" width="2.5" minGap="3.0" maxSpeed="6.94" accel="1.2" decel="3.5" sigma="0.5"
           guiShape="truck" color="0.6,0.4,0.2"/>

    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- ═══ PHASE 1: 0-90s — Heavy S outbound (0.7), moderate N (0.3), light E/W (0.1) ═══ -->

    <!-- South: 0.7 veh/s (strong outbound exodus) — 72% through, 10% left, 18% right -->
    <flow id="S1_through" route="S_through" begin="0" end="90" probability="0.504" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_left"    route="S_left"    begin="0" end="90" probability="0.070" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_right"   route="S_right"   begin="0" end="90" probability="0.126" departLane="best" departSpeed="desired" type="car"/>

    <!-- North: 0.3 veh/s (moderate inbound) -->
    <flow id="N1_through" route="N_through" begin="0" end="90" probability="0.216" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_left"    route="N_left"    begin="0" end="90" probability="0.042" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_right"   route="N_right"   begin="0" end="90" probability="0.042" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.1 veh/s -->
    <flow id="E1_through" route="E_through" begin="0" end="90" probability="0.072" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_left"    route="E_left"    begin="0" end="90" probability="0.014" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_right"   route="E_right"   begin="0" end="90" probability="0.014" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.1 veh/s -->
    <flow id="W1_through" route="W_through" begin="0" end="90" probability="0.072" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_left"    route="W_left"    begin="0" end="90" probability="0.014" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_right"   route="W_right"   begin="0" end="90" probability="0.014" departLane="best" departSpeed="desired" type="car"/>

    <!-- ═══ PHASE 2: 90-180s — Rush subsides, demand equalizes ═══ -->

    <!-- South drops to 0.25 -->
    <flow id="S2_through" route="S_through" begin="90" end="180" probability="0.175" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_left"    route="S_left"    begin="90" end="180" probability="0.038" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_right"   route="S_right"   begin="90" end="180" probability="0.038" departLane="best" departSpeed="desired" type="car"/>

    <!-- North stays 0.25 -->
    <flow id="N2_through" route="N_through" begin="90" end="180" probability="0.175" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_left"    route="N_left"    begin="90" end="180" probability="0.038" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_right"   route="N_right"   begin="90" end="180" probability="0.038" departLane="best" departSpeed="desired" type="car"/>

    <!-- E/W rise to 0.3 (nightlife district traffic picks up) -->
    <flow id="E2_through" route="E_through" begin="90" end="180" probability="0.210" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_left"    route="E_left"    begin="90" end="180" probability="0.045" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_right"   route="E_right"   begin="90" end="180" probability="0.045" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.3 veh/s -->
    <flow id="W2_through" route="W_through" begin="90" end="180" probability="0.210" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_left"    route="W_left"    begin="90" end="180" probability="0.045" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_right"   route="W_right"   begin="90" end="180" probability="0.045" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v4/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="180"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v4/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/adaptive_timer/v5/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v5/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T19:12:32.235581-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v5/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v5/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v5: School Zone Pulse (Suburban USA)                 -->
    <!-- Low baseline flow with sudden 30s burst from one direction   -->
    <!-- when school lets out — tests response to demand spikes       -->
    <!-- Phase 1 (0-60s):    Low baseline all directions 0.1 veh/s   -->
    <!-- Phase 2 (60-90s):   North SURGE 0.8 veh/s (school pickup)   -->
    <!-- Phase 3 (90-180s):  Returns to baseline                     -->
    <!-- Based on: FHWA School Zone traffic patterns, suburban data   -->
    <!-- Turn ratio: 55% through, 18% left, 27% right (residential)  -->
    <!-- ============================================================ -->

    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33" accel="2.0" decel="4.5" sigma="0.5"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="suv"   length="5.5"  width="2.2" minGap="2.5" maxSpeed="8.33" accel="1.8" decel="4.0" sigma="0.5"
           guiShape="passenger/sedan" color="0.3,0.3,0.7"/>

    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- ═══ PHASE 1: 0-60s — Low baseline everywhere (0.1 veh/s per approach) ═══ -->
    <flow id="N1_through" route="N_through" begin="0" end="60" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_left"    route="N_left"    begin="0" end="60" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_right"   route="N_right"   begin="0" end="60" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="S1_through" route="S_through" begin="0" end="60" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_left"    route="S_left"    begin="0" end="60" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_right"   route="S_right"   begin="0" end="60" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="E1_through" route="E_through" begin="0" end="60" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_left"    route="E_left"    begin="0" end="60" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_right"   route="E_right"   begin="0" end="60" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="W1_through" route="W_through" begin="0" end="60" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_left"    route="W_left"    begin="0" end="60" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_right"   route="W_right"   begin="0" end="60" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <!-- ═══ PHASE 2: 60-90s — North SURGES to 0.8 (school lets out) ═══ -->
    <flow id="N2_through" route="N_through" begin="60" end="90" probability="0.44" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_left"    route="N_left"    begin="60" end="90" probability="0.144" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_right"   route="N_right"   begin="60" end="90" probability="0.216" departLane="best" departSpeed="desired" type="car"/>

    <!-- Others stay at baseline -->
    <flow id="S2_through" route="S_through" begin="60" end="90" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_left"    route="S_left"    begin="60" end="90" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_right"   route="S_right"   begin="60" end="90" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="E2_through" route="E_through" begin="60" end="90" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_left"    route="E_left"    begin="60" end="90" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_right"   route="E_right"   begin="60" end="90" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="W2_through" route="W_through" begin="60" end="90" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_left"    route="W_left"    begin="60" end="90" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_right"   route="W_right"   begin="60" end="90" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <!-- ═══ PHASE 3: 90-180s — Returns to baseline ═══ -->
    <flow id="N3_through" route="N_through" begin="90" end="180" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N3_left"    route="N_left"    begin="90" end="180" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N3_right"   route="N_right"   begin="90" end="180" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="S3_through" route="S_through" begin="90" end="180" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S3_left"    route="S_left"    begin="90" end="180" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S3_right"   route="S_right"   begin="90" end="180" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="E3_through" route="E_through" begin="90" end="180" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E3_left"    route="E_left"    begin="90" end="180" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E3_right"   route="E_right"   begin="90" end="180" probability="0.027" departLane="best" departSpeed="desired" type="car"/>

    <flow id="W3_through" route="W_through" begin="90" end="180" probability="0.055" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W3_left"    route="W_left"    begin="90" end="180" probability="0.018" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W3_right"   route="W_right"   begin="90" end="180" probability="0.027" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v5/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="180"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v5/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/adaptive_timer/v6/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v6/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T19:12:32.388993-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v6/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v6/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v6: Oversaturated Arterial (Mumbai/Delhi-style)     -->
    <!-- All approaches near capacity — tests gridlock prevention     -->
    <!-- High demand on all 4 approaches simultaneously               -->
    <!-- N/S: 0.7 veh/s, E/W: 0.6 veh/s (v/c ratio ~0.9)           -->
    <!-- Turn ratio: 68% through, 8% left, 24% right                 -->
    <!-- Based on: Indian metro traffic data, HCM oversaturated ops   -->
    <!-- Duration: 180s                                               -->
    <!-- ============================================================ -->

    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33" accel="2.0" decel="4.5" sigma="0.5"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="auto"  length="3.5"  width="1.5" minGap="1.5" maxSpeed="6.94" accel="2.5" decel="5.0" sigma="0.7"
           guiShape="passenger/sedan" color="0.9,0.9,0.2"/>
    <vType id="bus"   length="12.0" width="2.5" minGap="3.0" maxSpeed="6.94" accel="1.2" decel="3.5" sigma="0.5"
           guiShape="bus" color="0.2,0.6,0.8"/>

    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- North: 0.7 veh/s (68% through, 8% left, 24% right) -->
    <flow id="N_through" route="N_through" begin="0" end="180" probability="0.476" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_left"    route="N_left"    begin="0" end="180" probability="0.056" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_right"   route="N_right"   begin="0" end="180" probability="0.168" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.7 veh/s -->
    <flow id="S_through" route="S_through" begin="0" end="180" probability="0.476" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_left"    route="S_left"    begin="0" end="180" probability="0.056" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_right"   route="S_right"   begin="0" end="180" probability="0.168" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.6 veh/s -->
    <flow id="E_through" route="E_through" begin="0" end="180" probability="0.408" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_left"    route="E_left"    begin="0" end="180" probability="0.048" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_right"   route="E_right"   begin="0" end="180" probability="0.144" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.6 veh/s -->
    <flow id="W_through" route="W_through" begin="0" end="180" probability="0.408" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_left"    route="W_left"    begin="0" end="180" probability="0.048" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_right"   route="W_right"   begin="0" end="180" probability="0.144" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v6/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="180"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v6/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/adaptive_timer/v7/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/adaptive_timer/v7/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T19:12:32.538833-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/adaptive_timer/v7/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v7/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v7: Stadium Event Dispersal (Special Event)         -->
    <!-- Models post-game/concert dispersal: one direction gets       -->
    <!-- extreme surge while others are nearly empty                  -->
    <!-- Phase 1 (0-30s):   Calm before dispersal (all 0.05 veh/s)  -->
    <!-- Phase 2 (30-120s): West SURGE 0.9 veh/s (stadium exit)      -->
    <!-- Phase 3 (120-180s): Gradual decay as crowd disperses        -->
    <!-- Based on: Event traffic management data, venue dispersal     -->
    <!-- Turn ratio during surge: 80% through (everyone heading out) -->
    <!-- ============================================================ -->

    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33" accel="2.0" decel="4.5" sigma="0.5"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="suv"   length="5.5"  width="2.2" minGap="2.5" maxSpeed="8.33" accel="1.8" decel="4.0" sigma="0.5"
           guiShape="passenger/sedan" color="0.2,0.2,0.6"/>

    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- ═══ PHASE 1: 0-30s — Calm before the storm (all 0.05 veh/s) ═══ -->
    <flow id="N1_through" route="N_through" begin="0" end="30" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_left"    route="N_left"    begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N1_right"   route="N_right"   begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="S1_through" route="S_through" begin="0" end="30" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_left"    route="S_left"    begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S1_right"   route="S_right"   begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="E1_through" route="E_through" begin="0" end="30" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_left"    route="E_left"    begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E1_right"   route="E_right"   begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="W1_through" route="W_through" begin="0" end="30" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_left"    route="W_left"    begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W1_right"   route="W_right"   begin="0" end="30" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <!-- ═══ PHASE 2: 30-120s — West SURGE 0.9 veh/s (80% through, 10% left, 10% right) ═══ -->
    <flow id="W2_through" route="W_through" begin="30" end="120" probability="0.72" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_left"    route="W_left"    begin="30" end="120" probability="0.09" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W2_right"   route="W_right"   begin="30" end="120" probability="0.09" departLane="best" departSpeed="desired" type="car"/>

    <!-- Others stay at 0.05 during surge -->
    <flow id="N2_through" route="N_through" begin="30" end="120" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_left"    route="N_left"    begin="30" end="120" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N2_right"   route="N_right"   begin="30" end="120" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="S2_through" route="S_through" begin="30" end="120" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_left"    route="S_left"    begin="30" end="120" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S2_right"   route="S_right"   begin="30" end="120" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="E2_through" route="E_through" begin="30" end="120" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_left"    route="E_left"    begin="30" end="120" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E2_right"   route="E_right"   begin="30" end="120" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <!-- ═══ PHASE 3: 120-180s — Gradual decay (West drops to 0.3, others stay 0.05) ═══ -->
    <flow id="W3_through" route="W_through" begin="120" end="180" probability="0.24" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W3_left"    route="W_left"    begin="120" end="180" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W3_right"   route="W_right"   begin="120" end="180" probability="0.03" departLane="best" departSpeed="desired" type="car"/>

    <flow id="N3_through" route="N_through" begin="120" end="180" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N3_left"    route="N_left"    begin="120" end="180" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N3_right"   route="N_right"   begin="120" end="180" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="S3_through" route="S_through" begin="120" end="180" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S3_left"    route="S_left"    begin="120" end="180" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S3_right"   route="S_right"   begin="120" end="180" probability="0.01" departLane="best" departSpeed="desired" type="car"/>

    <flow id="E3_through" route="E_through" begin="120" end="180" probability="0.03" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E3_left"    route="E_left"    begin="120" end="180" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E3_right"   route="E_right"   begin="120" end="180" probability="0.01" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/adaptive_timer/v7/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="180"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/adaptive_timer/v7/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/car_sprites/Topdown_vehicle_sprites_pack/Read Me.txt
================================================
Created By UnLuckY Studio

You are free to use it in your commercial projects. If you have any suggestions or questions, you can log on to www.unluckystudio.com




================================================
FILE: data/sumo_simulation/fixed_timer/v1/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/fixed_timer/v1/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T18:08:29.494855-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id=":C_5_0" index="0" speed="6.51" length="9.03" shape="110.40,104.80 107.95,105.15 106.20,106.20 105.15,107.95 104.80,110.40"/>
    </edge>
    <edge id=":C_6" function="internal">
        <lane id=":C_6_0" index="0" speed="8.33" length="20.80" shape="110.40,104.80 89.60,104.80"/>
        <lane id=":C_6_1" index="1" speed="8.33" length="20.80" shape="110.40,101.60 89.60,101.60"/>
    </edge>
    <edge id=":C_8" function="internal">
        <lane id=":C_8_0" index="0" speed="8.33" length="5.01" shape="110.40,101.60 105.44,100.89"/>
    </edge>
    <edge id=":C_9" function="internal">
        <lane id=":C_9_0" index="0" speed="3.65" length="1.44" shape="110.40,101.60 109.20,100.80"/>
    </edge>
    <edge id=":C_22" function="internal">
        <lane id=":C_22_0" index="0" speed="8.33" length="14.34" shape="105.44,100.89 105.15,100.85 101.40,98.60 99.15,94.85 98.40,89.60"/>
    </edge>
    <edge id=":C_23" function="internal">
        <lane id=":C_23_0" index="0" speed="3.65" length="3.23" shape="109.20,100.80 108.80,100.00 109.20,99.20 110.40,98.40"/>
    </edge>
    <edge id=":C_10" function="internal">
        <lane id=":C_10_0" index="0" speed="6.51" length="9.03" shape="104.80,89.60 105.15,92.05 106.20,93.80 107.95,94.85 110.40,95.20"/>
    </edge>
    <edge id=":C_11" function="internal">
        <lane id=":C_11_0" index="0" speed="8.33" length="20.80" shape="104.80,89.60 104.80,110.40"/>
        <lane id=":C_11_1" index="1" speed="8.33" length="20.80" shape="101.60,89.60 101.60,110.40"/>
    </edge>
    <edge id=":C_13" function="internal">
        <lane id=":C_13_0" index="0" speed="8.33" length="5.01" shape="101.60,89.60 100.89,94.56"/>
    </edge>
    <edge id=":C_14" function="internal">
        <lane id=":C_14_0" index="0" speed="3.65" length="1.44" shape="101.60,89.60 100.80,90.80"/>
    </edge>
    <edge id=":C_24" function="internal">
        <lane id=":C_24_0" index="0" speed="8.33" length="14.34" shape="100.89,94.56 100.85,94.85 98.60,98.60 94.85,100.85 89.60,101.60"/>
    </edge>
    <edge id=":C_25" function="internal">
        <lane id=":C_25_0" index="0" speed="3.65" length="3.23" shape="100.80,90.80 100.00,91.20 99.20,90.80 98.40,89.60"/>
    </edge>
    <edge id=":C_15" function="internal">
        <lane id=":C_15_0" index="0" speed="6.51" length="9.03" shape="89.60,95.20 92.05,94.85 93.80,93.80 94.85,92.05 95.20,89.60"/>
    </edge>
    <edge id=":C_16" function="internal">
        <lane id=":C_16_0" index="0" speed="8.33" length="20.80" shape="89.60,95.20 110.40,95.20"/>
        <lane id=":C_16_1" index="1" speed="8.33" length="20.80" shape="89.60,98.40 110.40,98.40"/>
    </edge>
    <edge id=":C_18" function="internal">
        <lane id=":C_18_0" index="0" speed="8.33" length="5.01" shape="89.60,98.40 94.56,99.11"/>
    </edge>
    <edge id=":C_19" function="internal">
        <lane id=":C_19_0" index="0" speed="3.65" length="1.44" shape="89.60,98.40 90.80,99.20"/>
    </edge>
    <edge id=":C_26" function="internal">
        <lane id=":C_26_0" index="0" speed="8.33" length="14.34" shape="94.56,99.11 94.85,99.15 98.60,101.40 100.85,105.15 101.60,110.40"/>
    </edge>
    <edge id=":C_27" function="internal">
        <lane id=":C_27_0" index="0" speed="3.65" length="3.23" shape="90.80,99.20 91.20,100.00 90.80,100.80 89.60,101.60"/>
    </edge>
    <edge id=":E_0" function="internal">
        <lane id=":E_0_0" index="0" speed="3.65" length="4.67" shape="200.00,98.40 201.20,99.20 201.60,100.00 201.20,100.80 200.00,101.60"/>
    </edge>
    <edge id=":N_0" function="internal">
        <lane id=":N_0_0" index="0" speed="3.65" length="4.67" shape="101.60,200.00 100.80,201.20 100.00,201.60 99.20,201.20 98.40,200.00"/>
    </edge>
    <edge id=":S_0" function="internal">
        <lane id=":S_0_0" index="0" speed="3.65" length="4.67" shape="98.40,-0.00 99.20,-1.20 100.00,-1.60 100.80,-1.20 101.60,0.00"/>
    </edge>
    <edge id=":W_0" function="internal">
        <lane id=":W_0_0" index="0" speed="3.65" length="4.67" shape="-0.00,101.60 -1.20,100.80 -1.60,100.00 -1.20,99.20 0.00,98.40"/>
    </edge>

    <edge id="C_to_E" from="C" to="E" priority="-1">
        <lane id="C_to_E_0" index="0" speed="8.33" length="89.60" shape="110.40,95.20 200.00,95.20"/>
        <lane id="C_to_E_1" index="1" speed="8.33" length="89.60" shape="110.40,98.40 200.00,98.40"/>
    </edge>
    <edge id="C_to_N" from="C" to="N" priority="-1">
        <lane id="C_to_N_0" index="0" speed="8.33" length="89.60" shape="104.80,110.40 104.80,200.00"/>
        <lane id="C_to_N_1" index="1" speed="8.33" length="89.60" shape="101.60,110.40 101.60,200.00"/>
    </edge>
    <edge id="C_to_S" from="C" to="S" priority="-1">
        <lane id="C_to_S_0" index="0" speed="8.33" length="89.60" shape="95.20,89.60 95.20,-0.00"/>
        <lane id="C_to_S_1" index="1" speed="8.33" length="89.60" shape="98.40,89.60 98.40,-0.00"/>
    </edge>
    <edge id="C_to_W" from="C" to="W" priority="-1">
        <lane id="C_to_W_0" index="0" speed="8.33" length="89.60" shape="89.60,104.80 -0.00,104.80"/>
        <lane id="C_to_W_1" index="1" speed="8.33" length="89.60" shape="89.60,101.60 -0.00,101.60"/>
    </edge>
    <edge id="E_to_C" from="E" to="C" priority="-1">
        <lane id="E_to_C_0" index="0" speed="8.33" length="89.60" shape="200.00,104.80 110.40,104.80"/>
        <lane id="E_to_C_1" index="1" speed="8.33" length="89.60" shape="200.00,101.60 110.40,101.60"/>
    </edge>
    <edge id="N_to_C" from="N" to="C" priority="-1">
        <lane id="N_to_C_0" index="0" speed="8.33" length="89.60" shape="95.20,200.00 95.20,110.40"/>
        <lane id="N_to_C_1" index="1" speed="8.33" length="89.60" shape="98.40,200.00 98.40,110.40"/>
    </edge>
    <edge id="S_to_C" from="S" to="C" priority="-1">
        <lane id="S_to_C_0" index="0" speed="8.33" length="89.60" shape="104.80,0.00 104.80,89.60"/>
        <lane id="S_to_C_1" index="1" speed="8.33" length="89.60" shape="101.60,0.00 101.60,89.60"/>
    </edge>
    <edge id="W_to_C" from="W" to="C" priority="-1">
        <lane id="W_to_C_0" index="0" speed="8.33" length="89.60" shape="0.00,95.20 89.60,95.20"/>
        <lane id="W_to_C_1" index="1" speed="8.33" length="89.60" shape="0.00,98.40 89.60,98.40"/>
    </edge>

    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="C" type="traffic_light" x="100.00" y="100.00" incLanes="N_to_C_0 N_to_C_1 E_to_C_0 E_to_C_1 S_to_C_0 S_to_C_1 W_to_C_0 W_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_20_0 :C_21_0 :C_5_0 :C_6_0 :C_6_1 :C_22_0 :C_23_0 :C_10_0 :C_11_0 :C_11_1 :C_24_0 :C_25_0 :C_15_0 :C_16_0 :C_16_1 :C_26_0 :C_27_0" shape="93.60,110.40 106.40,110.40 106.84,108.18 107.40,107.40 108.18,106.84 109.18,106.51 110.40,106.40 110.40,93.60 108.18,93.16 107.40,92.60 106.84,91.82 106.51,90.82 106.40,89.60 93.60,89.60 93.16,91.82 92.60,92.60 91.82,93.16 90.82,93.49 89.60,93.60 89.60,106.40 91.82,106.84 92.60,107.40 93.16,108.18 93.49,109.18">
        <request index="0"  response="00000000000000000000" foes="00000000000011000000" cont="0"/>
        <request index="1"  response="01000000000000000000" foes="01111110000111000000" cont="0"/>
        <request index="2"  response="01000000000100000000" foes="01111110000111000000" cont="0"/>
        <request index="3"  response="01000001100100000000" foes="01110001101111000000" cont="1"/>
        <request index="4"  response="01000001100000000000" foes="01000001100000000000" cont="1"/>
        <request index="5"  response="00000001100000000000" foes="00000001100000000000" cont="0"/>
        <request index="6"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="7"  response="00000011100000001111" foes="11000011100000001111" cont="0"/>
        <request index="8"  response="00110011100000001110" foes="00110111100000001110" cont="1"/>
        <request index="9"  response="00110000000000001000" foes="00110000000000001000" cont="1"/>
        <request index="10" response="00000000000000000000" foes="00110000000000000000" cont="0"/>
        <request index="11" response="00000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="12" response="01000000000100000000" foes="01110000000111111000" cont="0"/>
        <request index="13" response="01000000000100000110" foes="11110000000111000110" cont="1"/>
        <request index="14" response="00000000000100000110" foes="00000000000100000110" cont="1"/>
        <request index="15" response="00000000000000000110" foes="00000000000000000110" cont="0"/>
        <request index="16" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="17" response="00000011110000001110" foes="00000011111100001110" cont="0"/>
        <request index="18" response="00000011100011001110" foes="00000011100011011110" cont="1"/>
        <request index="19" response="00000010000011000000" foes="00000010000011000000" cont="1"/>
    </junction>
    <junction id="E" type="priority" x="200.00" y="100.00" incLanes="C_to_E_0 C_to_E_1" intLanes=":E_0_0" shape="200.00,100.00 200.00,93.60 200.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="N" type="priority" x="100.00" y="200.00" incLanes="C_to_N_0 C_to_N_1" intLanes=":N_0_0" shape="100.00,200.00 106.40,200.00 100.00,200.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="S" type="priority" x="100.00" y="0.00" incLanes="C_to_S_0 C_to_S_1" intLanes=":S_0_0" shape="100.00,-0.00 93.60,-0.00 100.00,-0.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>
    <junction id="W" type="priority" x="0.00" y="100.00" incLanes="C_to_W_0 C_to_W_1" intLanes=":W_0_0" shape="-0.00,100.00 -0.00,106.40 -0.00,100.00">
        <request index="0" response="0" foes="0" cont="0"/>
    </junction>

    <junction id=":C_20_0" type="internal" x="99.11" y="105.44" incLanes=":C_3_0 S_to_C_0 S_to_C_1" intLanes=":C_6_0 :C_6_1 :C_8_0 :C_9_0 :C_10_0 :C_11_0 :C_11_1 :C_16_0 :C_16_1 :C_18_0"/>
    <junction id=":C_21_0" type="internal" x="99.20" y="109.20" incLanes=":C_4_0 E_to_C_0 S_to_C_0 S_to_C_1 W_to_C_1" intLanes=":C_5_0 :C_11_0 :C_11_1 :C_18_0"/>
    <junction id=":C_22_0" type="internal" x="105.44" y="100.89" incLanes=":C_8_0 W_to_C_0 W_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_11_0 :C_11_1 :C_13_0 :C_14_0 :C_15_0 :C_16_0 :C_16_1"/>
    <junction id=":C_23_0" type="internal" x="109.20" y="100.80" incLanes=":C_9_0 N_to_C_1 S_to_C_0 W_to_C_0 W_to_C_1" intLanes=":C_3_0 :C_10_0 :C_16_0 :C_16_1"/>
    <junction id=":C_24_0" type="internal" x="100.89" y="94.56" incLanes=":C_13_0 N_to_C_0 N_to_C_1" intLanes=":C_0_0 :C_1_0 :C_1_1 :C_6_0 :C_6_1 :C_8_0 :C_16_0 :C_16_1 :C_18_0 :C_19_0"/>
    <junction id=":C_25_0" type="internal" x="100.80" y="90.80" incLanes=":C_14_0 E_to_C_1 N_to_C_0 N_to_C_1 W_to_C_0" intLanes=":C_1_0 :C_1_1 :C_8_0 :C_15_0"/>
    <junction id=":C_26_0" type="internal" x="94.56" y="99.11" incLanes=":C_18_0 E_to_C_0 E_to_C_1" intLanes=":C_1_0 :C_1_1 :C_3_0 :C_4_0 :C_5_0 :C_6_0 :C_6_1 :C_11_0 :C_11_1 :C_13_0"/>
    <junction id=":C_27_0" type="internal" x="90.80" y="99.20" incLanes=":C_19_0 E_to_C_0 E_to_C_1 N_to_C_0 S_to_C_1" intLanes=":C_0_0 :C_6_0 :C_6_1 :C_13_0"/>

    <connection from="C_to_E" to="E_to_C" fromLane="1" toLane="1" via=":E_0_0" dir="t" state="M"/>
    <connection from="C_to_N" to="N_to_C" fromLane="1" toLane="1" via=":N_0_0" dir="t" state="M"/>
    <connection from="C_to_S" to="S_to_C" fromLane="1" toLane="1" via=":S_0_0" dir="t" state="M"/>
    <connection from="C_to_W" to="W_to_C" fromLane="1" toLane="1" via=":W_0_0" dir="t" state="M"/>
    <connection from="E_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_5_0" tl="C" linkIndex="5" dir="r" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_6_0" tl="C" linkIndex="6" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_6_1" tl="C" linkIndex="7" dir="s" state="o"/>
    <connection from="E_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_8_0" tl="C" linkIndex="8" dir="l" state="o"/>
    <connection from="E_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_9_0" tl="C" linkIndex="9" dir="t" state="o"/>
    <connection from="N_to_C" to="C_to_W" fromLane="0" toLane="0" via=":C_0_0" tl="C" linkIndex="0" dir="r" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_1_0" tl="C" linkIndex="1" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_1_1" tl="C" linkIndex="2" dir="s" state="O"/>
    <connection from="N_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_3_0" tl="C" linkIndex="3" dir="l" state="o"/>
    <connection from="N_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_4_0" tl="C" linkIndex="4" dir="t" state="o"/>
    <connection from="S_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_10_0" tl="C" linkIndex="10" dir="r" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="0" toLane="0" via=":C_11_0" tl="C" linkIndex="11" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_11_1" tl="C" linkIndex="12" dir="s" state="O"/>
    <connection from="S_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_13_0" tl="C" linkIndex="13" dir="l" state="o"/>
    <connection from="S_to_C" to="C_to_S" fromLane="1" toLane="1" via=":C_14_0" tl="C" linkIndex="14" dir="t" state="o"/>
    <connection from="W_to_C" to="C_to_S" fromLane="0" toLane="0" via=":C_15_0" tl="C" linkIndex="15" dir="r" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="0" toLane="0" via=":C_16_0" tl="C" linkIndex="16" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_E" fromLane="1" toLane="1" via=":C_16_1" tl="C" linkIndex="17" dir="s" state="o"/>
    <connection from="W_to_C" to="C_to_N" fromLane="1" toLane="1" via=":C_18_0" tl="C" linkIndex="18" dir="l" state="o"/>
    <connection from="W_to_C" to="C_to_W" fromLane="1" toLane="1" via=":C_19_0" tl="C" linkIndex="19" dir="t" state="o"/>

    <connection from=":C_0" to="C_to_W" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_1" to="C_to_S" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_3" to="C_to_E" fromLane="0" toLane="1" via=":C_20_0" dir="l" state="m"/>
    <connection from=":C_20" to="C_to_E" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_4" to="C_to_N" fromLane="0" toLane="1" via=":C_21_0" dir="t" state="m"/>
    <connection from=":C_21" to="C_to_N" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_5" to="C_to_N" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_6" to="C_to_W" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_8" to="C_to_S" fromLane="0" toLane="1" via=":C_22_0" dir="l" state="m"/>
    <connection from=":C_22" to="C_to_S" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_9" to="C_to_E" fromLane="0" toLane="1" via=":C_23_0" dir="t" state="m"/>
    <connection from=":C_23" to="C_to_E" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_10" to="C_to_E" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_11" to="C_to_N" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_13" to="C_to_W" fromLane="0" toLane="1" via=":C_24_0" dir="l" state="m"/>
    <connection from=":C_24" to="C_to_W" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_14" to="C_to_S" fromLane="0" toLane="1" via=":C_25_0" dir="t" state="m"/>
    <connection from=":C_25" to="C_to_S" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":C_15" to="C_to_S" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":C_16" to="C_to_E" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":C_18" to="C_to_N" fromLane="0" toLane="1" via=":C_26_0" dir="l" state="m"/>
    <connection from=":C_26" to="C_to_N" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":C_19" to="C_to_W" fromLane="0" toLane="1" via=":C_27_0" dir="t" state="m"/>
    <connection from=":C_27" to="C_to_W" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":E_0" to="E_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":N_0" to="N_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":S_0" to="S_to_C" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":W_0" to="W_to_C" fromLane="0" toLane="1" dir="t" state="M"/>

</net>



================================================
FILE: data/sumo_simulation/fixed_timer/v1/intersection.nod.xml
================================================
<nodes>
    <!-- Center intersection -->
    <node id="C" x="0" y="0" type="traffic_light"/>

    <!-- Entry/exit endpoints (100m from center) -->
    <node id="N" x="0"    y="100"  type="priority"/>
    <node id="S" x="0"    y="-100" type="priority"/>
    <node id="E" x="100"  y="0"    type="priority"/>
    <node id="W" x="-100" y="0"    type="priority"/>
</nodes>



================================================
FILE: data/sumo_simulation/fixed_timer/v1/intersection.rou.xml
================================================
<routes>
    <!-- ============================================================ -->
    <!-- SCENARIO v1: Baseline traffic demand                         -->
    <!-- N/S: 0.3 veh/s,  E/W: 0.2 veh/s                            -->
    <!-- Turn split: 60% through, 20% left, 20% right                -->
    <!-- Duration: 120s                                               -->
    <!-- Fixed seed for reproducibility                               -->
    <!-- ============================================================ -->

    <!-- Vehicle types -->
    <vType id="car"   length="5.0"  width="2.0" minGap="2.5" maxSpeed="8.33"  accel="2.0" decel="4.5" sigma="0.5" probability="0.70"
           guiShape="passenger" color="0.8,0.1,0.1"/>
    <vType id="van"   length="5.75" width="2.2" minGap="2.5" maxSpeed="7.50"  accel="1.8" decel="4.0" sigma="0.5" probability="0.15"
           guiShape="delivery" color="0.9,0.8,0.3"/>
    <vType id="truck" length="10.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.10"
           guiShape="truck" color="0.6,0.4,0.2"/>
    <vType id="bus"   length="12.0" width="2.5" minGap="3.0" maxSpeed="6.94"  accel="1.2" decel="3.5" sigma="0.5" probability="0.05"
           guiShape="bus" color="0.2,0.6,0.8"/>

    <!-- Routes -->
    <route id="N_through" edges="N_to_C C_to_S"/>
    <route id="N_left"    edges="N_to_C C_to_E"/>
    <route id="N_right"   edges="N_to_C C_to_W"/>
    <route id="S_through" edges="S_to_C C_to_N"/>
    <route id="S_left"    edges="S_to_C C_to_W"/>
    <route id="S_right"   edges="S_to_C C_to_E"/>
    <route id="E_through" edges="E_to_C C_to_W"/>
    <route id="E_left"    edges="E_to_C C_to_S"/>
    <route id="E_right"   edges="E_to_C C_to_N"/>
    <route id="W_through" edges="W_to_C C_to_E"/>
    <route id="W_left"    edges="W_to_C C_to_N"/>
    <route id="W_right"   edges="W_to_C C_to_S"/>

    <!-- North: 0.3 veh/s -->
    <flow id="N_f_through" route="N_through" begin="0" end="120" probability="0.18" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_f_left"    route="N_left"    begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="N_f_right"   route="N_right"   begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>

    <!-- South: 0.3 veh/s -->
    <flow id="S_f_through" route="S_through" begin="0" end="120" probability="0.18" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_f_left"    route="S_left"    begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>
    <flow id="S_f_right"   route="S_right"   begin="0" end="120" probability="0.06" departLane="best" departSpeed="desired" type="car"/>

    <!-- East: 0.2 veh/s -->
    <flow id="E_f_through" route="E_through" begin="0" end="120" probability="0.12" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_f_left"    route="E_left"    begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
    <flow id="E_f_right"   route="E_right"   begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>

    <!-- West: 0.2 veh/s -->
    <flow id="W_f_through" route="W_through" begin="0" end="120" probability="0.12" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_f_left"    route="W_left"    begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
    <flow id="W_f_right"   route="W_right"   begin="0" end="120" probability="0.04" departLane="best" departSpeed="desired" type="car"/>
</routes>



================================================
FILE: data/sumo_simulation/fixed_timer/v1/intersection.sumocfg
================================================
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="120"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="42"/>
    </random_number>
    <gui_only>
        <gui-settings-file value="intersection.view.xml"/>
    </gui_only>
</configuration>



================================================
FILE: data/sumo_simulation/fixed_timer/v1/intersection.view.xml
================================================
<viewsettings>
    <scheme name="real world">
        <background backgroundColor="0.60,0.75,0.45" showGrid="0"/>
        <edges laneShowBorders="1" showLinkDecals="1" showLinkRules="1"
               edgeColorer="uniform" laneEdgeColor="0.30,0.30,0.30"/>
        <vehicles vehicleMode="0" vehicleQuality="3" vehicle_minSize="2.5"
                  vehicleName_show="0" vehicleName_size="40"/>
        <junctions junctionMode="0" junctionSize="1"
                   drawLinkTLIndex="0" drawLinkJunctionIndex="0"
                   junctionName_show="0"
                   tlsPhaseIndex_show="0"
                   showLane2Lane="0"/>
    </scheme>
    <delay value="50"/>
</viewsettings>



================================================
FILE: data/sumo_simulation/fixed_timer/v1/run_sumo_video.py
================================================
"""
SUMO GUI Video Capture — Fixed Timer v1
========================================
Runs sumo-gui and captures frames via TraCI screenshots.
Uses SUMO's native rendering — proper vehicle shapes, traffic lights, roads.

Usage:
    python run_sumo_video.py
"""

import os
import time
import shutil
import random
import traci
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = os.path.join(SCRIPT_DIR, "intersection.sumocfg")
OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "intersection_sim.mp4")

WIDTH = 1920
HEIGHT = 1080
FPS = 30
STEP_LENGTH = 1.0 / FPS  # one simulation step = one video frame

# Car colors for variety — assigned randomly via TraCI at spawn
CAR_COLORS = [
    (200, 30, 30, 255),    # red
    (30, 70, 180, 255),    # blue
    (230, 230, 230, 255),  # white
    (40, 40, 40, 255),     # black
    (180, 180, 180, 255),  # silver
    (10, 130, 50, 255),    # green
    (220, 180, 50, 255),   # gold
    (80, 80, 80, 255),     # dark gray
    (170, 80, 40, 255),    # brown
    (100, 50, 150, 255),   # purple
]


def main():
    sumo_gui = shutil.which("sumo-gui")
    if not sumo_gui:
        raise FileNotFoundError("sumo-gui not found")
    print(f"Using: {sumo_gui}")

    sumo_cmd = [
        sumo_gui,
        "-c", SUMO_CFG,
        "--start",
        "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0",
        "--gui-testing",
        "--step-length", str(STEP_LENGTH),
        "--gui-settings-file", os.path.join(SCRIPT_DIR, "intersection.view.xml"),
    ]

    traci.start(sumo_cmd)
    print("sumo-gui started.")

    # Wait for GUI to fully initialize
    for _ in range(10):
        traci.simulationStep()

    # Set view — network center is at (100,100)
    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 100, 100)
    traci.gui.setZoom("View #0", 350)

    # Let view settle
    for _ in range(5):
        traci.simulationStep()

    screenshot_dir = os.path.join(SCRIPT_DIR, "_frames")
    os.makedirs(screenshot_dir, exist_ok=True)

    frame_count = 0
    colored_vehicles = set()

    print("Capturing frames...")

    # Phase 1: Run simulation and request all screenshots
    frame_paths = []
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Assign random colors to new vehicles
            for vid in traci.vehicle.getIDList():
                if vid not in colored_vehicles:
                    color = random.choice(CAR_COLORS)
                    traci.vehicle.setColor(vid, color)
                    colored_vehicles.add(vid)

            # Request screenshot — SUMO writes it before the NEXT step completes
            fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
            traci.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
            frame_paths.append(fpath)
            frame_count += 1

            if frame_count % 300 == 0:
                n = traci.vehicle.getIDCount()
                sim_time = traci.simulation.getTime()
                print(f"  Frame {frame_count}, sim time: {sim_time:.1f}s, vehicles: {n}")

    except traci.exceptions.FatalTraCIError:
        print("Simulation ended.")
    finally:
        # One more step to flush the last screenshot
        try:
            traci.simulationStep()
        except Exception:
            pass
        traci.close()

    print(f"Captured {frame_count} frames. Encoding video...")

    # Phase 2: Encode all frames to video at constant FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
    written = 0

    for fpath in frame_paths:
        # Wait briefly if file not yet written
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

    # Cleanup frame images
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


if __name__ == "__main__":
    main()



================================================
FILE: data/sumo_simulation/fixed_timer/v2/intersection.edg.xml
================================================
<edges>
    <!-- Inbound edges (toward intersection) — 30 km/h = 8.33 m/s -->
    <edge id="N_to_C" from="N" to="C" numLanes="2" speed="8.33"/>
    <edge id="S_to_C" from="S" to="C" numLanes="2" speed="8.33"/>
    <edge id="E_to_C" from="E" to="C" numLanes="2" speed="8.33"/>
    <edge id="W_to_C" from="W" to="C" numLanes="2" speed="8.33"/>

    <!-- Outbound edges (leaving intersection) -->
    <edge id="C_to_N" from="C" to="N" numLanes="2" speed="8.33"/>
    <edge id="C_to_S" from="C" to="S" numLanes="2" speed="8.33"/>
    <edge id="C_to_E" from="C" to="E" numLanes="2" speed="8.33"/>
    <edge id="C_to_W" from="C" to="W" numLanes="2" speed="8.33"/>
</edges>



================================================
FILE: data/sumo_simulation/fixed_timer/v2/intersection.net.xml
================================================
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2026-06-02T18:45:42.966121-06:00 by Eclipse SUMO netconvert 1.26.0
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <node-files value="intersection.nod.xml"/>
        <edge-files value="intersection.edg.xml"/>
    </input>

    <output>
        <output-file value="intersection.net.xml"/>
    </output>

</netconvertConfiguration>
-->

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>

    <edge id=":C_0" function="internal">
        <lane id=":C_0_0" index="0" speed="6.51" length="9.03" shape="95.20,110.40 94.85,107.95 93.80,106.20 92.05,105.15 89.60,104.80"/>
    </edge>
    <edge id=":C_1" function="internal">
        <lane id=":C_1_0" index="0" speed="8.33" length="20.80" shape="95.20,110.40 95.20,89.60"/>
        <lane id=":C_1_1" index="1" speed="8.33" length="20.80" shape="98.40,110.40 98.40,89.60"/>
    </edge>
    <edge id=":C_3" function="internal">
        <lane id=":C_3_0" index="0" speed="8.33" length="5.01" shape="98.40,110.40 99.11,105.44"/>
    </edge>
    <edge id=":C_4" function="internal">
        <lane id=":C_4_0" index="0" speed="3.65" length="1.44" shape="98.40,110.40 99.20,109.20"/>
    </edge>
    <edge id=":C_20" function="internal">
        <lane id=":C_20_0" index="0" speed="8.33" length="14.34" shape="99.11,105.44 99.15,105.15 101.40,101.40 105.15,99.15 110.40,98.40"/>
    </edge>
    <edge id=":C_21" function="internal">
        <lane id=":C_21_0" index="0" speed="3.65" length="3.23" shape="99.20,109.20 100.00,108.80 100.80,109.20 101.60,110.40"/>
    </edge>
    <edge id=":C_5" function="internal">
        <lane id="