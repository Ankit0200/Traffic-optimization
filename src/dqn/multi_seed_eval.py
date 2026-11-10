"""
Multi-Seed Evaluation with Statistical Tests
=============================================
Runs Fixed, DQN (per-version), and Adaptive across multiple seeds
for each scenario. Reports mean ± std and p-values.

Usage:
    python multi_seed_eval.py                        # 5 seeds, all versions
    python multi_seed_eval.py --seeds 10             # 10 seeds
    python multi_seed_eval.py --versions v3 v5       # specific versions
"""

import os
import sys
import json
import shutil
import argparse
import uuid
import numpy as np
from pathlib import Path
from scipy import stats
import traci

SCRIPT_DIR = Path(__file__).resolve().parent           # src/dqn/
PROJECT_ROOT = SCRIPT_DIR.parents[1]                   # Research_zhou/
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_simulation"   # SUMO configs live here
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SUMO_DIR))

# ── Constants ────────────────────────────────────────────────────────────
STEP_LENGTH = 1.0 / 30
FPS = 30
MIN_GREEN = 10.0
YELLOW_DUR = 3.0
DECISION_INTERVAL = 5.0
STEPS_PER_DECISION = int(DECISION_INTERVAL / STEP_LENGTH)

NS_GREEN_STATE  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW_STATE = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN_STATE  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW_STATE = "rrrrryyyyyrrrrryyyyy"

EDGE_IDS = ["N_to_C", "S_to_C", "E_to_C", "W_to_C"]

# Grid / vision constants (from optimize_weights.py)
ARRIVAL_LOOKAHEAD = 30.0


# ── Fixed Timer ──────────────────────────────────────────────────────────

def run_fixed(version, seed):
    sumo_bin = shutil.which("sumo")
    cfg = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
    label = f"fixed_{uuid.uuid4().hex[:8]}"

    sumo_cmd = [sumo_bin, "-c", cfg, "--step-length", str(STEP_LENGTH),
                "--no-warnings", "--no-step-log", "--seed", str(seed),
                "--end", "300", "--time-to-teleport", "120",
                "--waiting-time-memory", "3600"]

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        traci.start(sumo_cmd, label=label, numRetries=60)
    conn = traci.getConnection(label)

    for _ in range(15):
        conn.simulationStep()

    conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
    phase = "NS_GREEN"
    timer = 0.0

    veh_last_wait = {}
    veh_total_wait = []
    arrived = 0

    while conn.simulation.getMinExpectedNumber() > 0:
        conn.simulationStep()
        timer += STEP_LENGTH

        for vid in conn.vehicle.getIDList():
            veh_last_wait[vid] = conn.vehicle.getAccumulatedWaitingTime(vid)
        for vid in conn.simulation.getArrivedIDList():
            if vid in veh_last_wait:
                veh_total_wait.append(veh_last_wait.pop(vid))
        arrived += conn.simulation.getArrivedNumber()

        if phase == "NS_GREEN" and timer >= 42.0:
            conn.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
            phase = "NS_YELLOW"; timer = 0.0
        elif phase == "NS_YELLOW" and timer >= YELLOW_DUR:
            conn.trafficlight.setRedYellowGreenState("C", EW_GREEN_STATE)
            phase = "EW_GREEN"; timer = 0.0
        elif phase == "EW_GREEN" and timer >= 42.0:
            conn.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
            phase = "EW_YELLOW"; timer = 0.0
        elif phase == "EW_YELLOW" and timer >= YELLOW_DUR:
            conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
            phase = "NS_GREEN"; timer = 0.0

    conn.close()
    # Include vehicles still in the network when simulation ended (survivorship bias fix)
    veh_total_wait.extend(veh_last_wait.values())
    arrived += len(veh_last_wait)
    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    return total_wait / max(arrived, 1)


# ── DQN (per-version) ───────────────────────────────────────────────────

def run_dqn(version, seed):
    from stable_baselines3 import DQN
    import gymnasium as gym
    from gymnasium import spaces

    model_path = str(SCRIPT_DIR / f"dqn_models/dqn_{version}.zip")
    if not os.path.exists(model_path):
        model_path = str(SCRIPT_DIR / "dqn_traffic_model.zip")
    model = DQN.load(model_path)

    sumo_bin = shutil.which("sumo")
    cfg = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
    label = f"dqn_{uuid.uuid4().hex[:8]}"

    sumo_cmd = [sumo_bin, "-c", cfg, "--step-length", str(STEP_LENGTH),
                "--no-warnings", "--no-step-log", "--seed", str(seed),
                "--end", "300", "--time-to-teleport", "120",
                "--waiting-time-memory", "3600"]

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        traci.start(sumo_cmd, label=label, numRetries=60)
    conn = traci.getConnection(label)

    for _ in range(15):
        conn.simulationStep()
    conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)

    current_phase = 0
    phase_timer = 0.0
    yellow_active = False
    yellow_timer = 0.0

    veh_last_wait = {}
    veh_total_wait = []
    arrived = 0

    while conn.simulation.getMinExpectedNumber() > 0:
        # Observe
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

        if action == 1 and not yellow_active and phase_timer >= MIN_GREEN:
            if current_phase == 0:
                conn.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
            else:
                conn.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
            yellow_active = True
            yellow_timer = 0.0

        for _ in range(STEPS_PER_DECISION):
            if conn.simulation.getMinExpectedNumber() <= 0:
                break
            conn.simulationStep()
            phase_timer += STEP_LENGTH

            for vid in conn.vehicle.getIDList():
                veh_last_wait[vid] = conn.vehicle.getAccumulatedWaitingTime(vid)
            for vid in conn.simulation.getArrivedIDList():
                if vid in veh_last_wait:
                    veh_total_wait.append(veh_last_wait.pop(vid))
            arrived += conn.simulation.getArrivedNumber()

            if yellow_active:
                yellow_timer += STEP_LENGTH
                if yellow_timer >= YELLOW_DUR:
                    current_phase = 1 - current_phase
                    if current_phase == 0:
                        conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
                    else:
                        conn.trafficlight.setRedYellowGreenState("C", EW_GREEN_STATE)
                    yellow_active = False
                    phase_timer = 0.0

    conn.close()
    # Include vehicles still in the network when simulation ended (survivorship bias fix)
    veh_total_wait.extend(veh_last_wait.values())
    arrived += len(veh_last_wait)
    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    return total_wait / max(arrived, 1)


# ── Adaptive (Optuna-optimized) ──────────────────────────────────────────

def run_adaptive(version, seed, use_vision=False):
    """Run adaptive controller with a specific seed.

    Args:
        use_vision: If True, use YOLO vision-based detection (sumo-gui)
                    instead of TraCI ground truth (headless sumo).
    """
    opt_path = SUMO_DIR / "optimized_params.json"
    with open(str(opt_path)) as f:
        opt = json.load(f)
    if opt.get("mode") == "per_version" and version in opt.get("per_version", {}):
        params = opt["per_version"][version]["best_params"]
    else:
        params = {"w_queue": 1.0, "w_arrival": 3.0, "w_through": 0.3, "cycle_total": 90.0}

    # Suppress LSTM/QueueManager discovery output
    import io, contextlib

    if use_vision:
        from optimize_weights import run_vision_adaptive
        with contextlib.redirect_stdout(io.StringIO()):
            return run_vision_adaptive(version, params, seed=seed)
    else:
        from optimize_weights import run_headless_adaptive
        with contextlib.redirect_stdout(io.StringIO()):
            return run_headless_adaptive(version, params, seed=seed)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation with statistical tests")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5)")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed value (default: 42)")
    parser.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5", "v6", "v7"],
                        help="Scenario versions")
    parser.add_argument("--vision", action="store_true",
                        help="Use YOLO vision-based detection for adaptive controller "
                             "(requires sumo-gui; slower but genuinely vision-based)")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    versions = args.versions

    use_vision = args.vision

    print(f"\n{'='*70}")
    print(f"  MULTI-SEED EVALUATION")
    print(f"  Seeds: {seeds}")
    print(f"  Versions: {versions}")
    print(f"  Adaptive mode: {'YOLO vision-based (sumo-gui)' if use_vision else 'TraCI ground truth (headless)'}")
    print(f"  Total runs: {len(seeds)} seeds × {len(versions)} versions × 3 methods = {len(seeds)*len(versions)*3}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Collect results: results[version][method] = [wait_per_seed]
    results = {}

    for v in versions:
        results[v] = {"fixed": [], "dqn": [], "adaptive": []}
        print(f"  ── {v} ──")
        sys.stdout.flush()

        for seed in seeds:
            print(f"    seed={seed}:", end=" ", flush=True)

            # Fixed
            w = run_fixed(v, seed)
            results[v]["fixed"].append(round(w, 2))
            print(f"fixed={w:.2f}", end=" ", flush=True)

            # DQN
            w = run_dqn(v, seed)
            results[v]["dqn"].append(round(w, 2))
            print(f"dqn={w:.2f}", end=" ", flush=True)

            # Adaptive
            w = run_adaptive(v, seed, use_vision=use_vision)
            results[v]["adaptive"].append(round(w, 2))
            print(f"adaptive={w:.2f}")
            sys.stdout.flush()

    # ── Compute statistics ──
    print(f"\n{'='*70}")
    print(f"  RESULTS (mean ± std over {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"\n  {'Version':<8} {'Fixed':>16} {'DQN-each':>16} {'Adaptive':>16}")
    print(f"  {'-'*60}")

    overall = {"fixed": [], "dqn": [], "adaptive": []}

    for v in versions:
        f = results[v]["fixed"]
        d = results[v]["dqn"]
        a = results[v]["adaptive"]
        overall["fixed"].extend(f)
        overall["dqn"].extend(d)
        overall["adaptive"].extend(a)

        print(f"  {v:<8} {np.mean(f):>7.2f} ± {np.std(f):<5.2f}"
              f"  {np.mean(d):>7.2f} ± {np.std(d):<5.2f}"
              f"  {np.mean(a):>7.2f} ± {np.std(a):<5.2f}")

    print(f"  {'-'*60}")
    print(f"  {'Overall':<8} {np.mean(overall['fixed']):>7.2f} ± {np.std(overall['fixed']):<5.2f}"
          f"  {np.mean(overall['dqn']):>7.2f} ± {np.std(overall['dqn']):<5.2f}"
          f"  {np.mean(overall['adaptive']):>7.2f} ± {np.std(overall['adaptive']):<5.2f}")

    # ── Statistical tests ──
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE")
    print(f"  (paired tests on all {len(seeds)} × {len(versions)} = {len(seeds)*len(versions)} observations)")
    print(f"{'='*70}\n")

    # Use ALL seed-level observations (n = seeds × versions) for proper power
    # Each observation is paired: same (version, seed) across methods
    all_obs = {"fixed": [], "dqn": [], "adaptive": []}
    for v in versions:
        for i in range(len(seeds)):
            all_obs["fixed"].append(results[v]["fixed"][i])
            all_obs["dqn"].append(results[v]["dqn"][i])
            all_obs["adaptive"].append(results[v]["adaptive"][i])

    # Also keep version means for the JSON output
    version_means = {"fixed": [], "dqn": [], "adaptive": []}
    for v in versions:
        for method in ["fixed", "dqn", "adaptive"]:
            version_means[method].append(np.mean(results[v][method]))

    comparisons = [
        ("Adaptive vs Fixed", "adaptive", "fixed"),
        ("Adaptive vs DQN", "adaptive", "dqn"),
        ("DQN vs Fixed", "dqn", "fixed"),
    ]

    for label, method_a, method_b in comparisons:
        a_vals = np.array(all_obs[method_a])
        b_vals = np.array(all_obs[method_b])
        diff_pct = (np.mean(b_vals) - np.mean(a_vals)) / np.mean(b_vals) * 100

        # Wilcoxon signed-rank test (paired, non-parametric, one-tailed)
        try:
            stat, p_value = stats.wilcoxon(b_vals, a_vals, alternative='greater')
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        except ValueError:
            p_value = float('nan')
            sig = "n/a"

        # Paired t-test (one-tailed to match Wilcoxon)
        t_stat, t_p_two = stats.ttest_rel(b_vals, a_vals)
        t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2  # one-tailed
        t_sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"

        print(f"  {label}:")
        print(f"    Improvement: {diff_pct:+.1f}%")
        print(f"    Wilcoxon (one-tailed): p={p_value:.6f} ({sig})")
        print(f"    Paired t-test (one-tailed): t={t_stat:.3f}, p={t_p:.6f} ({t_sig})")
        print()

    print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # ── Save results ──
    output = {
        "config": {
            "seeds": seeds,
            "versions": versions,
            "num_seeds": len(seeds),
        },
        "per_version": {},
        "overall": {
            "fixed": {"mean": round(np.mean(overall["fixed"]), 2),
                      "std": round(np.std(overall["fixed"]), 2)},
            "dqn": {"mean": round(np.mean(overall["dqn"]), 2),
                    "std": round(np.std(overall["dqn"]), 2)},
            "adaptive": {"mean": round(np.mean(overall["adaptive"]), 2),
                         "std": round(np.std(overall["adaptive"]), 2)},
        },
        "statistical_tests": {},
    }

    for v in versions:
        output["per_version"][v] = {
            "fixed": {"values": results[v]["fixed"],
                      "mean": round(np.mean(results[v]["fixed"]), 2),
                      "std": round(np.std(results[v]["fixed"]), 2)},
            "dqn": {"values": results[v]["dqn"],
                    "mean": round(np.mean(results[v]["dqn"]), 2),
                    "std": round(np.std(results[v]["dqn"]), 2)},
            "adaptive": {"values": results[v]["adaptive"],
                         "mean": round(np.mean(results[v]["adaptive"]), 2),
                         "std": round(np.std(results[v]["adaptive"]), 2)},
        }

    for label, method_a, method_b in comparisons:
        a_vals = np.array(all_obs[method_a])
        b_vals = np.array(all_obs[method_b])
        try:
            _, w_p = stats.wilcoxon(b_vals, a_vals, alternative='greater')
        except ValueError:
            w_p = float('nan')
        t_stat, t_p_two = stats.ttest_rel(b_vals, a_vals)
        t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
        output["statistical_tests"][label] = {
            "n_observations": len(a_vals),
            "improvement_pct": round((np.mean(b_vals) - np.mean(a_vals)) / np.mean(b_vals) * 100, 2),
            "wilcoxon_p": round(w_p, 6),
            "ttest_p": round(t_p, 6),
            "ttest_t": round(t_stat, 3),
        }

    out_path = SCRIPT_DIR / "multi_seed_results.json"
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
