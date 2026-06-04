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
