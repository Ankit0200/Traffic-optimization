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
