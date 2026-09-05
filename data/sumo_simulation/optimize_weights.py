"""
Optuna Hyperparameter Optimization for Adaptive Signal Control
==============================================================
Finds optimal W_QUEUE, W_ARRIVAL, W_THROUGH, and CYCLE_TOTAL by running
headless SUMO simulations across all scenarios (v1-v7) and minimizing
average waiting time.

Usage:
    python optimize_weights.py                  # 100 trials, all scenarios
    python optimize_weights.py --n_trials 50    # fewer trials
    python optimize_weights.py --versions v3 v7 # specific scenarios only
"""

import os
import sys
import json
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

import traci
import optuna

# ── Project paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from queue_manager import QueueManager
from arrival_rate import ArrivalRateEstimator
from lstm_predictor import RealtimePredictor

# ── Constants (not optimized) ────────────────────────────────────────────
STEP_LENGTH = 1.0 / 30
FPS = 30
WIDTH, HEIGHT, CELL_SIZE = 1920, 1080, 50
MIN_GREEN = 10.0
MAX_GREEN = 60.0
YELLOW_DUR = 3.0

NS_GREEN_STATE  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW_STATE = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN_STATE  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW_STATE = "rrrrryyyyyrrrrryyyyy"

EDGE_TO_DIR = {"N_to_C": "NB", "S_to_C": "SB", "E_to_C": "EB", "W_to_C": "WB"}
NS_EDGES = ["N_to_C", "S_to_C"]
EW_EDGES = ["E_to_C", "W_to_C"]
NS_DIRS = {"NB", "SB"}
EW_DIRS = {"EB", "WB"}

# Default intersection geometry params
DEFAULT_INTERSECTION_DIST = 30.0
DEFAULT_INTERSECTION_MARGIN = 0.35


# ── Grid mapper (same as compare.py) ─────────────────────────────────────

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


# ── Scoring functions (parameterized) ────────────────────────────────────

def compute_demand_score(queue_count, arrival_rate, through_ratio, params):
    lookahead = 10.0  # fixed
    projected = arrival_rate * lookahead
    bonus = 1.0 + params["w_through"] * through_ratio
    return (params["w_queue"] * queue_count + params["w_arrival"] * projected) * bonus

def compute_green_split(ns_score, ew_score, cycle_total):
    total = ns_score + ew_score
    if total < 0.01:
        return cycle_total / 2, cycle_total / 2
    ns = max(MIN_GREEN, min(MAX_GREEN, cycle_total * ns_score / total))
    ew = max(MIN_GREEN, min(MAX_GREEN, cycle_total * ew_score / total))
    return ns, ew


def compute_scores(queue_mgr, arrival_est, lstm, vehicle_cells,
                   vehicle_entry_dir, approach_to_dir, active_cells,
                   frame_count, params):
    """Compute demand scores and green split — silent version."""
    # 1. Queue counts
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

    # 2. Arrival rates
    ns_arrival = ew_arrival = 0.0
    all_rates = arrival_est.compute_all_rates(frame_count)
    for label, rate in all_rates.items():
        d = approach_to_dir.get(label)
        if d in NS_DIRS:   ns_arrival += rate
        elif d in EW_DIRS: ew_arrival += rate

    # 3. LSTM through-ratios
    ns_tr, ew_tr = 0.5, 0.5
    ns_thr, ns_tot, ew_thr, ew_tot = 0, 0, 0, 0
    if lstm:
        for vid, cells in vehicle_cells.items():
            if vid not in active_cells:
                continue
            entry = vehicle_entry_dir.get(vid)
            if not entry:
                continue
            pred, conf, _ = lstm.predict(cells)
            if pred is None or conf < 0.4:
                continue
            through = lstm.is_through_movement(entry, pred)
            if entry in NS_DIRS:
                ns_tot += 1
                if through: ns_thr += 1
            elif entry in EW_DIRS:
                ew_tot += 1
                if through: ew_thr += 1
        if ns_tot > 0: ns_tr = ns_thr / ns_tot
        if ew_tot > 0: ew_tr = ew_thr / ew_tot

    # 4. Compute scores
    ns_score = compute_demand_score(ns_queue, ns_arrival, ns_tr, params)
    ew_score = compute_demand_score(ew_queue, ew_arrival, ew_tr, params)
    ns_green, ew_green = compute_green_split(ns_score, ew_score, params["cycle_total"])

    return ns_score, ew_score, ns_green, ew_green


# ── Headless adaptive simulation ─────────────────────────────────────────

# Cache LSTM models so we don't reload every trial
_lstm_cache = {}
_traj_cache = {}


def run_headless_adaptive(version, params):
    """
    Run one headless SUMO simulation with the given adaptive parameters.
    Returns avg_waiting_time_per_vehicle (float).
    """
    sumo_bin = shutil.which("sumo")
    if not sumo_bin:
        raise RuntimeError("sumo binary not found on PATH")

    cfg = str(SCRIPT_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"No config for {version}: {cfg}")

    # LSTM model (cached)
    lstm_path = str(PROJECT_ROOT / "models" / f"fixed_{version}_trajectories_lstm_model.pt")
    if lstm_path not in _lstm_cache:
        _lstm_cache[lstm_path] = RealtimePredictor(lstm_path) if os.path.exists(lstm_path) else None
    lstm = _lstm_cache[lstm_path]

    # Trajectory data for QueueManager (cached)
    traj_path = str(PROJECT_ROOT / "data" / "transitions" / "sumo_transition" / f"fixed_{version}_trajectories.json")
    if traj_path not in _traj_cache:
        if os.path.exists(traj_path):
            with open(traj_path) as f:
                _traj_cache[traj_path] = json.load(f)
        else:
            _traj_cache[traj_path] = None
    traj_data = _traj_cache[traj_path]

    # Launch headless SUMO
    sumo_cmd = [
        sumo_bin, "-c", cfg,
        "--step-length", str(STEP_LENGTH),
        "--no-warnings",
        "--no-step-log",
    ]
    traci.start(sumo_cmd)

    try:
        # Warm up
        for _ in range(15):
            traci.simulationStep()

        net_bounds = traci.simulation.getNetBoundary()
        mapper = GridMapper(net_bounds)

        # Setup QueueManager
        queue_mgr = None
        approach_to_dir = {}
        if traj_data:
            grid_w = WIDTH // CELL_SIZE + 1
            grid_h = HEIGHT // CELL_SIZE + 1
            queue_mgr = QueueManager(grid_w, grid_h, fps=FPS,
                                     intersection_dist=DEFAULT_INTERSECTION_DIST,
                                     intersection_margin=DEFAULT_INTERSECTION_MARGIN)
            queue_mgr.auto_discover_approaches(traj_data["trajectories"], traj_data["grid_cols"])
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
        traci.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)

        current_phase = "NS_GREEN"
        phase_timer = 0.0
        ns_green_dur = ew_green_dur = params["cycle_total"] / 2

        vehicle_cells = defaultdict(list)
        vehicle_entry_dir = {}
        prev_cells = {}
        prev_vids = set()
        frame_count = 0

        total_waiting = 0.0
        departed = arrived = 0
        veh_last_wait = {}
        veh_total_wait = []

        # Main loop
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

            for vid in traci.simulation.getArrivedIDList():
                if vid in veh_last_wait:
                    veh_total_wait.append(veh_last_wait.pop(vid))

            for vid in prev_vids - current_vids:
                if queue_mgr: queue_mgr.vehicle_exited(vid)
                arrival_est.vehicle_exited(vid)
            prev_vids = current_vids

            n = traci.vehicle.getIDCount()
            total_waiting += n
            departed += traci.simulation.getDepartedNumber()
            arrived += traci.simulation.getArrivedNumber()

            # Phase transitions
            if current_phase == "NS_GREEN" and phase_timer >= ns_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
                current_phase = "NS_YELLOW"
                phase_timer = 0.0

            elif current_phase == "NS_YELLOW" and phase_timer >= YELLOW_DUR:
                _, _, ns_green_dur, ew_green_dur = compute_scores(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, params)
                traci.trafficlight.setRedYellowGreenState("C", EW_GREEN_STATE)
                current_phase = "EW_GREEN"
                phase_timer = 0.0

            elif current_phase == "EW_GREEN" and phase_timer >= ew_green_dur:
                traci.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
                current_phase = "EW_YELLOW"
                phase_timer = 0.0

            elif current_phase == "EW_YELLOW" and phase_timer >= YELLOW_DUR:
                _, _, ns_green_dur, ew_green_dur = compute_scores(
                    queue_mgr, arrival_est, lstm, vehicle_cells,
                    vehicle_entry_dir, approach_to_dir, active_cells,
                    frame_count, params)
                traci.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
                current_phase = "NS_GREEN"
                phase_timer = 0.0

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try:
            traci.simulationStep()
        except:
            pass
        traci.close()

    # Compute result
    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    arrived_count = max(arrived, 1)
    avg_wait = total_wait / arrived_count

    return avg_wait


# ── Optuna objective ─────────────────────────────────────────────────────

def make_objective(versions):
    """Create an Optuna objective function for the given scenario versions."""

    def objective(trial):
        params = {
            "w_queue":     trial.suggest_float("w_queue",     0.1, 5.0),
            "w_arrival":   trial.suggest_float("w_arrival",   0.1, 10.0),
            "w_through":   trial.suggest_float("w_through",   0.0, 1.0),
            "cycle_total": trial.suggest_float("cycle_total", 30.0, 150.0),
        }

        total_avg_wait = 0.0
        failed = 0

        for version in versions:
            try:
                avg_wait = run_headless_adaptive(version, params)
                total_avg_wait += avg_wait
            except Exception as e:
                print(f"    [WARN] {version} failed: {e}")
                total_avg_wait += 100.0  # penalty for failed sim
                failed += 1

        mean_wait = total_avg_wait / len(versions)

        print(f"  Trial {trial.number}: "
              f"w_q={params['w_queue']:.2f} w_a={params['w_arrival']:.2f} "
              f"w_t={params['w_through']:.2f} cycle={params['cycle_total']:.1f} "
              f"→ avg_wait={mean_wait:.2f}s"
              f"{f' ({failed} failed)' if failed else ''}")

        return mean_wait

    return objective


# ── Main ─────────────────────────────────────────────────────────────────

def optimize_single_version(version, n_trials, baseline):
    """Optimize parameters for a single version/scenario."""
    print(f"\n  ── Optimizing {version} ({n_trials} trials) ──")

    # Baseline for this version
    try:
        base_wait = run_headless_adaptive(version, baseline)
        print(f"    Baseline: {base_wait:.2f}s")
    except Exception as e:
        print(f"    Baseline FAILED: {e}")
        return None

    def objective(trial):
        params = {
            "w_queue":     trial.suggest_float("w_queue",     0.1, 5.0),
            "w_arrival":   trial.suggest_float("w_arrival",   0.1, 10.0),
            "w_through":   trial.suggest_float("w_through",   0.0, 1.0),
            "cycle_total": trial.suggest_float("cycle_total", 30.0, 150.0),
        }
        avg_wait = run_headless_adaptive(version, params)
        print(f"    Trial {trial.number}: "
              f"Q={params['w_queue']:.2f} A={params['w_arrival']:.2f} "
              f"T={params['w_through']:.2f} C={params['cycle_total']:.0f} "
              f"→ {avg_wait:.2f}s")
        sys.stdout.flush()
        return avg_wait

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    improvement = (base_wait - best.value) / base_wait * 100 if base_wait > 0 else 0
    print(f"    Best: {best.value:.2f}s (improvement: {improvement:+.1f}%)")
    print(f"    Params: Q={best.params['w_queue']:.2f} A={best.params['w_arrival']:.2f} "
          f"T={best.params['w_through']:.2f} C={best.params['cycle_total']:.0f}s")
    sys.stdout.flush()

    return {
        "best_params": best.params,
        "best_avg_wait": best.value,
        "baseline_avg_wait": base_wait,
        "improvement_pct": round(improvement, 2),
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize adaptive signal weights with Optuna")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5", "v6", "v7"],
                        help="Scenario versions to optimize across (default: v2-v7)")
    parser.add_argument("--mode", choices=["global", "per_version"], default="per_version",
                        help="'global' = one param set for all versions, 'per_version' = optimize each separately (default: per_version)")
    parser.add_argument("--study_name", default="adaptive_signal_opt", help="Optuna study name")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Check which versions actually have configs
    valid_versions = []
    for v in args.versions:
        cfg = SCRIPT_DIR / f"adaptive_timer/{v}/intersection.sumocfg"
        if cfg.exists():
            valid_versions.append(v)
        else:
            print(f"[WARN] Skipping {v} — no config found at {cfg}")

    if not valid_versions:
        print("No valid versions found. Exiting.")
        return

    baseline = {"w_queue": 1.0, "w_arrival": 3.0, "w_through": 0.3, "cycle_total": 90.0}

    print(f"\n{'='*60}")
    print(f"  OPTUNA OPTIMIZATION — {args.mode.upper()} MODE")
    print(f"  Scenarios: {valid_versions}")
    print(f"  Trials: {args.n_trials}" + (f" per version" if args.mode == "per_version" else ""))
    print(f"{'='*60}")
    sys.stdout.flush()

    if args.mode == "per_version":
        # ── Per-version optimization ────────────────────────────────
        per_version_results = {}

        for v in valid_versions:
            result = optimize_single_version(v, args.n_trials, baseline)
            if result:
                per_version_results[v] = result

        # Save results
        output = {
            "mode": "per_version",
            "baseline_params": baseline,
            "per_version": per_version_results,
        }

        out_path = SCRIPT_DIR / "optimized_params.json"
        with open(str(out_path), "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {out_path}")

        # Summary table
        print(f"\n{'='*60}")
        print(f"  PER-VERSION OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"  {'Version':<10} {'Baseline':>10} {'Optimized':>10} {'Change':>10}  {'Best Params'}")
        print(f"  {'-'*75}")
        for v, r in per_version_results.items():
            bp = r["best_params"]
            param_str = f"Q={bp['w_queue']:.1f} A={bp['w_arrival']:.1f} T={bp['w_through']:.2f} C={bp['cycle_total']:.0f}s"
            print(f"  {v:<10} {r['baseline_avg_wait']:>9.2f}s {r['best_avg_wait']:>9.2f}s "
                  f"{r['improvement_pct']:>+9.1f}%  {param_str}")
        sys.stdout.flush()

    else:
        # ── Global optimization (original behavior) ─────────────────
        print(f"\n  Current params: {baseline}")

        print(f"\n  Running baseline across all scenarios...")
        baseline_total = 0.0
        for v in valid_versions:
            try:
                w = run_headless_adaptive(v, baseline)
                print(f"    {v}: {w:.2f}s avg wait")
                baseline_total += w
            except Exception as e:
                print(f"    {v}: FAILED ({e})")
                baseline_total += 100.0
        baseline_mean = baseline_total / len(valid_versions)
        print(f"  Baseline mean avg wait: {baseline_mean:.2f}s")

        print(f"\n{'='*60}")
        print(f"  Starting optimization...")
        print(f"{'='*60}\n")

        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.enqueue_trial(baseline)

        objective = make_objective(valid_versions)
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        best = study.best_trial
        improvement = (baseline_mean - best.value) / baseline_mean * 100

        print(f"\n{'='*60}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"\n  Best trial: #{best.number}")
        print(f"  Best avg wait: {best.value:.2f}s (baseline: {baseline_mean:.2f}s)")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"\n  Best parameters:")
        for k, v in best.params.items():
            old = baseline[k]
            print(f"    {k}: {old:.2f} → {v:.2f}")

        output = {
            "mode": "global",
            "best_params": best.params,
            "best_avg_wait": best.value,
            "baseline_params": baseline,
            "baseline_avg_wait": baseline_mean,
            "improvement_pct": round(improvement, 2),
            "n_trials": args.n_trials,
            "versions": valid_versions,
            "all_trials": [
                {"number": t.number, "params": t.params, "value": t.value}
                for t in study.trials if t.value is not None
            ],
        }

        out_path = SCRIPT_DIR / "optimized_params.json"
        with open(str(out_path), "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {out_path}")

        print(f"\n  Per-scenario comparison (best params vs baseline):")
        print(f"  {'Scenario':<12} {'Baseline':>10} {'Optimized':>10} {'Change':>10}")
        print(f"  {'-'*42}")
        for v in valid_versions:
            try:
                w_opt = run_headless_adaptive(v, best.params)
                w_base = run_headless_adaptive(v, baseline)
                delta = (w_base - w_opt) / w_base * 100 if w_base > 0 else 0
                print(f"  {v:<12} {w_base:>10.2f}s {w_opt:>10.2f}s {delta:>+9.1f}%")
            except Exception as e:
                print(f"  {v:<12} ERROR: {e}")


if __name__ == "__main__":
    main()
