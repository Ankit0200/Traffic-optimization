"""
Independent Verification of Traffic Simulation Metrics
======================================================
Standalone script that re-implements Fixed and DQN control loops from
scratch using traci directly, tracking every vehicle individually to
verify that multi_seed_eval.py computes correct averages.

Scenario: v3, seed: 42
"""

import os
import sys
import shutil
import uuid
from pathlib import Path

import traci
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_simulation"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SUMO_DIR))

# ── Constants ───────────────────────────────────────────────────────────
VERSION = "v3"
SEED = 42
STEP_LENGTH = 1.0 / 30
MIN_GREEN = 10.0
YELLOW_DUR = 3.0
DECISION_INTERVAL = 5.0
STEPS_PER_DECISION = int(DECISION_INTERVAL / STEP_LENGTH)

NS_GREEN  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW = "rrrrryyyyyrrrrryyyyy"

EDGE_IDS = ["N_to_C", "S_to_C", "E_to_C", "W_to_C"]

CFG = str(SUMO_DIR / f"adaptive_timer/{VERSION}/intersection.sumocfg")
SUMO_BIN = shutil.which("sumo")

def base_cmd():
    return [SUMO_BIN, "-c", CFG,
            "--step-length", str(STEP_LENGTH),
            "--no-warnings", "--no-step-log",
            "--seed", str(SEED),
            "--end", "300",
            "--time-to-teleport", "120",
            "--waiting-time-memory", "3600"]


def print_vehicle_details(label, arrived_waits, stranded_waits, all_seen):
    """Print detailed per-vehicle accounting."""
    n_arrived = len(arrived_waits)
    n_stranded = len(stranded_waits)
    n_total = n_arrived + n_stranded
    sum_arrived = sum(arrived_waits)
    sum_stranded = sum(stranded_waits)
    total_wait = sum_arrived + sum_stranded
    avg = total_wait / max(n_total, 1)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Vehicles that ENTERED simulation (all seen): {len(all_seen)}")
    print(f"  Vehicles that ARRIVED (completed trip):      {n_arrived}")
    print(f"  Vehicles STRANDED (still in network at end): {n_stranded}")
    print(f"  Arrived + Stranded = Total:                  {n_arrived} + {n_stranded} = {n_total}")
    print(f"  Match all-seen?                              {'YES' if n_total == len(all_seen) else 'NO *** MISMATCH ***'}")
    print()
    print(f"  Sum of wait times (arrived vehicles):  {sum_arrived:.4f}")
    print(f"  Sum of wait times (stranded vehicles): {sum_stranded:.4f}")
    print(f"  Total wait (all vehicles):             {total_wait:.4f}")
    print(f"  Average wait = total / n_total:        {total_wait:.4f} / {n_total} = {avg:.4f}")
    print()

    # Show sample of per-vehicle waits
    all_waits = sorted(arrived_waits + stranded_waits)
    if len(all_waits) <= 20:
        print(f"  All per-vehicle waits: {[round(w,2) for w in all_waits]}")
    else:
        print(f"  First 10 per-vehicle waits: {[round(w,2) for w in all_waits[:10]]}")
        print(f"  Last  10 per-vehicle waits: {[round(w,2) for w in all_waits[-10:]]}")

    print(f"  Min wait: {min(all_waits):.4f}   Max wait: {max(all_waits):.4f}   Median: {np.median(all_waits):.4f}")
    return avg


# ═══════════════════════════════════════════════════════════════════════
#  FIXED TIMER — independent implementation
# ═══════════════════════════════════════════════════════════════════════

def verify_fixed():
    label = f"vfy_fixed_{uuid.uuid4().hex[:8]}"
    traci.start(base_cmd(), label=label, numRetries=60)
    conn = traci.getConnection(label)

    # Warm-up: 15 steps
    for _ in range(15):
        conn.simulationStep()

    conn.trafficlight.setRedYellowGreenState("C", NS_GREEN)
    phase = "NS_GREEN"
    timer = 0.0

    # Independent tracking
    all_seen = set()          # every vehicle ID we ever saw
    veh_last_wait = {}        # vid -> last known accumulated wait
    arrived_waits = []        # wait times of completed vehicles
    stranded_waits_dict = {}  # will be filled at end
    arrived_count_sumo = 0    # SUMO's arrived counter for cross-check

    step_count = 0
    while conn.simulation.getMinExpectedNumber() > 0:
        conn.simulationStep()
        step_count += 1
        timer += STEP_LENGTH

        # Track every vehicle currently in the network
        current_ids = conn.vehicle.getIDList()
        for vid in current_ids:
            all_seen.add(vid)
            veh_last_wait[vid] = conn.vehicle.getAccumulatedWaitingTime(vid)

        # Capture arrived vehicles
        arrived_ids = conn.simulation.getArrivedIDList()
        arrived_count_sumo += conn.simulation.getArrivedNumber()
        for vid in arrived_ids:
            all_seen.add(vid)  # should already be there
            if vid in veh_last_wait:
                arrived_waits.append(veh_last_wait.pop(vid))
            else:
                # Vehicle arrived but we never tracked it — shouldn't happen
                arrived_waits.append(0.0)
                print(f"  WARNING: {vid} arrived but was never tracked!")

        # Signal control: fixed 42s green, 3s yellow
        if phase == "NS_GREEN" and timer >= 42.0:
            conn.trafficlight.setRedYellowGreenState("C", NS_YELLOW)
            phase = "NS_YELLOW"; timer = 0.0
        elif phase == "NS_YELLOW" and timer >= YELLOW_DUR:
            conn.trafficlight.setRedYellowGreenState("C", EW_GREEN)
            phase = "EW_GREEN"; timer = 0.0
        elif phase == "EW_GREEN" and timer >= 42.0:
            conn.trafficlight.setRedYellowGreenState("C", EW_YELLOW)
            phase = "EW_YELLOW"; timer = 0.0
        elif phase == "EW_YELLOW" and timer >= YELLOW_DUR:
            conn.trafficlight.setRedYellowGreenState("C", NS_GREEN)
            phase = "NS_GREEN"; timer = 0.0

    conn.close()

    # Stranded vehicles: still in veh_last_wait
    stranded_waits = list(veh_last_wait.values())

    print(f"\n  [Fixed] SUMO arrived counter: {arrived_count_sumo}")
    print(f"  [Fixed] Our arrived list len: {len(arrived_waits)}")
    print(f"  [Fixed] Stranded count:       {len(stranded_waits)}")
    print(f"  [Fixed] Total sim steps:      {step_count}")

    avg = print_vehicle_details("FIXED TIMER (42s/3s yellow)", arrived_waits, stranded_waits, all_seen)
    return avg


# ═══════════════════════════════════════════════════════════════════════
#  DQN — independent implementation
# ═══════════════════════════════════════════════════════════════════════

def verify_dqn():
    from stable_baselines3 import DQN

    model_path = str(SCRIPT_DIR / f"dqn_models/dqn_{VERSION}.zip")
    if not os.path.exists(model_path):
        model_path = str(SCRIPT_DIR / "dqn_traffic_model.zip")
    print(f"\n  Loading DQN model from: {model_path}")
    model = DQN.load(model_path)

    label = f"vfy_dqn_{uuid.uuid4().hex[:8]}"
    traci.start(base_cmd(), label=label, numRetries=60)
    conn = traci.getConnection(label)

    # Warm-up: 15 steps
    for _ in range(15):
        conn.simulationStep()
    conn.trafficlight.setRedYellowGreenState("C", NS_GREEN)

    current_phase = 0  # 0=NS, 1=EW
    phase_timer = 0.0
    yellow_active = False
    yellow_timer = 0.0

    # Independent tracking
    all_seen = set()
    veh_last_wait = {}
    arrived_waits = []
    arrived_count_sumo = 0

    step_count = 0
    decision_count = 0

    while conn.simulation.getMinExpectedNumber() > 0:
        # Build observation
        queues = []
        speeds = []
        for edge in EDGE_IDS:
            queues.append(float(conn.edge.getLastStepHaltingNumber(edge)))
            speeds.append(float(conn.edge.getLastStepMeanSpeed(edge)))

        obs = np.array(
            queues + speeds + [float(current_phase), phase_timer / 60.0],
            dtype=np.float32,
        )

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        decision_count += 1

        if action == 1 and not yellow_active and phase_timer >= MIN_GREEN:
            if current_phase == 0:
                conn.trafficlight.setRedYellowGreenState("C", NS_YELLOW)
            else:
                conn.trafficlight.setRedYellowGreenState("C", EW_YELLOW)
            yellow_active = True
            yellow_timer = 0.0

        for _ in range(STEPS_PER_DECISION):
            if conn.simulation.getMinExpectedNumber() <= 0:
                break
            conn.simulationStep()
            step_count += 1
            phase_timer += STEP_LENGTH

            # Track vehicles
            current_ids = conn.vehicle.getIDList()
            for vid in current_ids:
                all_seen.add(vid)
                veh_last_wait[vid] = conn.vehicle.getAccumulatedWaitingTime(vid)

            arrived_ids = conn.simulation.getArrivedIDList()
            arrived_count_sumo += conn.simulation.getArrivedNumber()
            for vid in arrived_ids:
                all_seen.add(vid)
                if vid in veh_last_wait:
                    arrived_waits.append(veh_last_wait.pop(vid))
                else:
                    arrived_waits.append(0.0)
                    print(f"  WARNING: {vid} arrived but was never tracked!")

            if yellow_active:
                yellow_timer += STEP_LENGTH
                if yellow_timer >= YELLOW_DUR:
                    current_phase = 1 - current_phase
                    if current_phase == 0:
                        conn.trafficlight.setRedYellowGreenState("C", NS_GREEN)
                    else:
                        conn.trafficlight.setRedYellowGreenState("C", EW_GREEN)
                    yellow_active = False
                    phase_timer = 0.0

    conn.close()

    stranded_waits = list(veh_last_wait.values())

    print(f"\n  [DQN] SUMO arrived counter: {arrived_count_sumo}")
    print(f"  [DQN] Our arrived list len: {len(arrived_waits)}")
    print(f"  [DQN] Stranded count:       {len(stranded_waits)}")
    print(f"  [DQN] Total sim steps:      {step_count}")
    print(f"  [DQN] Total decisions:      {decision_count}")

    avg = print_vehicle_details("DQN CONTROLLER", arrived_waits, stranded_waits, all_seen)
    return avg


# ═══════════════════════════════════════════════════════════════════════
#  ADAPTIVE — just call existing code for comparison
# ═══════════════════════════════════════════════════════════════════════

def get_adaptive_reference():
    """Call the existing adaptive code for reference comparison."""
    import json
    from optimize_weights import run_headless_adaptive

    opt_path = SUMO_DIR / "optimized_params.json"
    with open(str(opt_path)) as f:
        opt = json.load(f)
    if opt.get("mode") == "per_version" and VERSION in opt.get("per_version", {}):
        params = opt["per_version"][VERSION]["best_params"]
    else:
        params = {"w_queue": 1.0, "w_arrival": 3.0, "w_through": 0.3, "cycle_total": 90.0}

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        avg = run_headless_adaptive(VERSION, params, seed=SEED)
    return avg


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  INDEPENDENT METRIC VERIFICATION")
    print(f"  Scenario: {VERSION}   Seed: {SEED}")
    print(f"  SUMO config: {CFG}")
    print(f"  Step length: {STEP_LENGTH:.6f}s  (1/30)")
    print("=" * 70)

    # Run fixed
    fixed_avg = verify_fixed()

    # Run DQN
    dqn_avg = verify_dqn()

    # Get adaptive reference
    print("\n  Running adaptive (existing code) for reference...")
    try:
        adaptive_avg = get_adaptive_reference()
        adaptive_str = f"{adaptive_avg:.4f}"
    except Exception as e:
        adaptive_avg = None
        adaptive_str = f"ERROR: {e}"

    # ── Comparison Table ──
    # Load expected values from multi_seed_results.json
    import json
    results_path = SCRIPT_DIR / "multi_seed_results.json"
    expected = {"fixed": None, "dqn": None, "adaptive": None}
    if results_path.exists():
        with open(str(results_path)) as f:
            data = json.load(f)
        v_data = data.get("per_version", {}).get(VERSION, {})
        for method in ["fixed", "dqn", "adaptive"]:
            vals = v_data.get(method, {}).get("values", [])
            if vals:
                expected[method] = vals[0]  # seed=42 is first seed

    print(f"\n\n{'='*70}")
    print(f"  COMPARISON TABLE — {VERSION} / seed={SEED}")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'Verified':>12} {'multi_seed_eval':>16} {'Delta':>10}")
    print(f"  {'-'*52}")

    for method, verified in [("Fixed", fixed_avg), ("DQN", dqn_avg), ("Adaptive", adaptive_avg)]:
        exp = expected[method.lower()]
        if verified is not None and exp is not None:
            delta = verified - exp
            print(f"  {method:<12} {verified:>12.4f} {exp:>16.2f} {delta:>+10.4f}")
        elif verified is not None:
            print(f"  {method:<12} {verified:>12.4f} {'N/A':>16} {'N/A':>10}")
        else:
            print(f"  {method:<12} {adaptive_str:>12} {exp if exp else 'N/A':>16} {'N/A':>10}")

    print(f"\n  NOTE: Delta should be ~0 if metrics match.")
    print(f"        Small floating-point differences (<0.01) are expected.")
    print(f"        Large differences indicate a bug in the evaluation code.")
    print(f"{'='*70}")
