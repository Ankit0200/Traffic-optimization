"""
DQN Live Demo — Run trained DQN with SUMO GUI, record video
============================================================
Runs a per-version DQN model with sumo-gui, captures every frame,
and encodes to mp4. Saves to dqn_timer/<version>/.

Usage:
    python dqn_live_demo.py                        # run all versions
    python dqn_live_demo.py --version v3           # single version
    python dqn_live_demo.py --version v3 --no-record  # watch live, no recording
"""

import os
import sys
import json
import shutil
import random
import argparse
import uuid
import numpy as np
import cv2
from pathlib import Path
from stable_baselines3 import DQN
import traci

SCRIPT_DIR = Path(__file__).resolve().parent           # src/dqn/
PROJECT_ROOT = SCRIPT_DIR.parents[1]                   # Research_zhou/
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_simulation"   # SUMO configs live here

# ── Constants ────────────────────────────────────────────────────────────
STEP_LENGTH = 1.0 / 30
FPS = 30
DECISION_INTERVAL = 5.0
STEPS_PER_DECISION = int(DECISION_INTERVAL / STEP_LENGTH)
MIN_GREEN = 10.0
YELLOW_DUR = 3.0
WIDTH, HEIGHT = 1280, 720

NS_GREEN_STATE  = "GGGggrrrrrGGGggrrrrr"
NS_YELLOW_STATE = "yyyyyrrrrryyyyyrrrrr"
EW_GREEN_STATE  = "rrrrrGGGggrrrrrGGGgg"
EW_YELLOW_STATE = "rrrrryyyyyrrrrryyyyy"

EDGE_IDS = ["N_to_C", "S_to_C", "E_to_C", "W_to_C"]

CAR_COLORS = [
    (255, 87, 51, 255), (255, 195, 0, 255), (46, 204, 113, 255),
    (52, 152, 219, 255), (155, 89, 182, 255), (241, 196, 15, 255),
    (230, 126, 34, 255), (231, 76, 60, 255), (26, 188, 156, 255),
    (52, 73, 94, 255), (22, 160, 133, 255), (192, 57, 43, 255),
]


def encode_video(frame_paths, output_path):
    print(f"  Encoding {len(frame_paths)} frames → {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    written = 0
    for fpath in frame_paths:
        for _ in range(10):
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                break
            import time; time.sleep(0.01)
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
    print(f"  Written: {written} frames, {written/FPS:.1f}s video")
    return written


def run_dqn_recorded(version, record=True):
    """Run trained DQN on a scenario with SUMO GUI, optionally record video."""

    # Load per-version model
    model_path = str(SCRIPT_DIR / f"dqn_models/dqn_{version}.zip")
    if not os.path.exists(model_path):
        model_path = str(SCRIPT_DIR / "dqn_traffic_model.zip")
    if not os.path.exists(model_path):
        print(f"  No trained model found. Run 'python dqn_baseline.py train' first.")
        return None

    model = DQN.load(model_path)

    # Set up output directory
    out_dir = SCRIPT_DIR / f"dqn_timer/{version}"
    os.makedirs(str(out_dir), exist_ok=True)

    output_video = str(out_dir / "intersection_sim.mp4")

    # Launch SUMO GUI
    sumo_gui = shutil.which("sumo-gui")
    assert sumo_gui, "sumo-gui not found on PATH"

    cfg_path = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
    view_xml = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.view.xml")
    conn_label = f"dqn_demo_{uuid.uuid4().hex[:8]}"

    sumo_cmd = [
        sumo_gui, "-c", cfg_path,
        "--start", "--quit-on-end",
        "--window-size", f"{WIDTH},{HEIGHT}",
        "--delay", "0", "--gui-testing",
        "--step-length", str(STEP_LENGTH),
    ]
    if os.path.exists(view_xml):
        sumo_cmd += ["--gui-settings-file", view_xml]

    print(f"\n[DQN {version}] Launching sumo-gui...")
    traci.start(sumo_cmd, label=conn_label)
    conn = traci.getConnection(conn_label)

    # Set up view
    for _ in range(10):
        conn.simulationStep()
    conn.gui.setSchema("View #0", "real world")
    conn.gui.setOffset("View #0", 100, 100)
    conn.gui.setZoom("View #0", 350)
    for _ in range(5):
        conn.simulationStep()

    # Take over TLS
    conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)

    # Screenshot directory
    screenshot_dir = str(out_dir / "_frames")
    os.makedirs(screenshot_dir, exist_ok=True)

    # State
    current_phase = 0  # 0=NS, 1=EW
    phase_timer = 0.0
    yellow_active = False
    yellow_timer = 0.0
    decision_count = 0
    frame_count = 0
    colored = set()
    frame_paths = []

    # Metrics
    veh_last_wait = {}
    veh_total_wait = []
    departed = arrived = max_veh = 0
    total_vehicle_steps = 0

    # Decision log
    decisions = []
    phase_names = ["NS_GREEN", "EW_GREEN"]

    print(f"[DQN {version}] Running DQN-controlled simulation...")

    try:
        while conn.simulation.getMinExpectedNumber() > 0:
            # ── DQN decision point (every DECISION_INTERVAL) ──
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

            action_str = "keep"
            switched = False

            if action == 1 and not yellow_active and phase_timer >= MIN_GREEN:
                if current_phase == 0:
                    conn.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
                else:
                    conn.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
                yellow_active = True
                yellow_timer = 0.0
                switched = True
                action_str = "SWITCH"
            elif action == 1 and phase_timer < MIN_GREEN:
                action_str = "switch(blocked)"

            sim_time = conn.simulation.getTime()
            decisions.append({
                "decision": decision_count,
                "sim_time": round(sim_time, 1),
                "action": action_str,
                "phase": phase_names[current_phase],
                "phase_timer": round(phase_timer, 1),
                "queues": {"N": int(queues[0]), "S": int(queues[1]),
                           "E": int(queues[2]), "W": int(queues[3])},
                "speeds": {"N": round(speeds[0], 2), "S": round(speeds[1], 2),
                           "E": round(speeds[2], 2), "W": round(speeds[3], 2)},
            })

            # ── Simulate DECISION_INTERVAL seconds, capture every frame ──
            for _ in range(STEPS_PER_DECISION):
                if conn.simulation.getMinExpectedNumber() <= 0:
                    break
                conn.simulationStep()
                frame_count += 1
                phase_timer += STEP_LENGTH

                # Color cars
                for vid in conn.vehicle.getIDList():
                    if vid not in colored:
                        conn.vehicle.setColor(vid, random.choice(CAR_COLORS))
                        colored.add(vid)
                    veh_last_wait[vid] = conn.vehicle.getAccumulatedWaitingTime(vid)

                for vid in conn.simulation.getArrivedIDList():
                    if vid in veh_last_wait:
                        veh_total_wait.append(veh_last_wait.pop(vid))

                n = conn.vehicle.getIDCount()
                total_vehicle_steps += n
                if n > max_veh:
                    max_veh = n
                departed += conn.simulation.getDepartedNumber()
                arrived += conn.simulation.getArrivedNumber()

                # Yellow → complete switch
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

                # Screenshot
                if record:
                    fpath = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.png")
                    conn.gui.screenshot("View #0", fpath, WIDTH, HEIGHT)
                    frame_paths.append(fpath)

            if decision_count % 5 == 0:
                print(f"  [DQN {version}] Decision #{decision_count}, t={sim_time:.0f}s, "
                      f"vehicles={conn.vehicle.getIDCount()}, frames={frame_count}")
                sys.stdout.flush()

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try:
            conn.simulationStep()
        except:
            pass
        conn.close()

    # Encode video
    if record and frame_paths:
        written = encode_video(frame_paths, output_video)
        try:
            os.rmdir(screenshot_dir)
        except:
            pass
    else:
        written = 0

    # Compute metrics
    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    avg_wait = total_wait / max(arrived, 1)

    metrics = {
        "version": version,
        "controller": "DQN (per-version)",
        "sim_time": round(frame_count * STEP_LENGTH, 1),
        "departed": departed,
        "arrived": arrived,
        "max_vehicles": max_veh,
        "total_wait_time": round(total_wait, 2),
        "avg_wait_per_vehicle": round(avg_wait, 2),
        "total_decisions": decision_count,
        "video_frames": written,
    }

    # Save metrics
    metrics_path = str(out_dir / "metrics_dqn.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save decision log
    log_path = str(out_dir / "decision_log.json")
    with open(log_path, "w") as f:
        json.dump({"version": version, "decisions": decisions}, f, indent=2)

    print(f"  [DQN {version}] Done — avg wait: {avg_wait:.2f}s, "
          f"video: {written/FPS:.1f}s, decisions: {decision_count}")
    print(f"  Saved: {output_video}")
    print(f"  Saved: {metrics_path}")
    print(f"  Saved: {log_path}")
    sys.stdout.flush()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN with SUMO GUI and record video")
    parser.add_argument("--version", default=None,
                        help="Scenario version (default: all v2-v7)")
    parser.add_argument("--no-record", action="store_true",
                        help="Watch live without recording video")
    args = parser.parse_args()

    versions = [args.version] if args.version else ["v2", "v3", "v4", "v5", "v6", "v7"]
    record = not args.no_record

    print(f"{'='*60}")
    print(f"  DQN LIVE DEMO — {'recording' if record else 'live only'}")
    print(f"  Versions: {versions}")
    print(f"  Output: dqn_timer/<version>/")
    print(f"{'='*60}")

    all_metrics = {}
    for v in versions:
        m = run_dqn_recorded(v, record=record)
        if m:
            all_metrics[v] = m

    if all_metrics:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"  {'Version':<10} {'Avg Wait':>10} {'Decisions':>10}")
        print(f"  {'-'*35}")
        for v, m in all_metrics.items():
            print(f"  {v:<10} {m['avg_wait_per_vehicle']:>9.2f}s {m['total_decisions']:>10}")
        print(f"{'='*60}")
