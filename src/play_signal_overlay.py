"""
Play video with signal phase overlay — no tracking, just visualization.

Usage:
    python play_signal_overlay.py \
        --video ../data/sumo_simulation/adaptive_timer/v2/intersection_sim.mp4 \
        --signal_log ../data/sumo_simulation/adaptive_timer/v2/decision_log.json

Controls:  SPACE = pause/resume,  q = quit
"""

import cv2
import json
import argparse

YELLOW_DUR = 3.0

def build_phase_timeline(signal_cycles):
    events = []
    if not signal_cycles:
        return events

    events.append((0, signal_cycles[0]["time"] - YELLOW_DUR, "NS_GREEN", None))
    events.append((signal_cycles[0]["time"] - YELLOW_DUR, signal_cycles[0]["time"], "NS_YELLOW", None))

    for i, cyc in enumerate(signal_cycles):
        is_ew = (i % 2 == 0)
        phase = "EW_GREEN" if is_ew else "NS_GREEN"
        if i + 1 < len(signal_cycles):
            next_t = signal_cycles[i + 1]["time"]
            events.append((cyc["time"], next_t - YELLOW_DUR, phase, cyc))
            yellow = "EW_YELLOW" if is_ew else "NS_YELLOW"
            events.append((next_t - YELLOW_DUR, next_t, yellow, None))
        else:
            green_dur = cyc["ew_green"] if is_ew else cyc["ns_green"]
            events.append((cyc["time"], cyc["time"] + green_dur, phase, cyc))
    return events


def get_phase(events, t):
    for start, end, phase, cyc in events:
        if start <= t < end:
            return phase, cyc, end - t
    return "NS_GREEN", None, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--signal_log", required=True)
    args = parser.parse_args()

    with open(args.signal_log) as f:
        signal_cycles = json.load(f)

    events = build_phase_timeline(signal_cycles)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    print(f"Video: {int(cap.get(3))}x{int(cap.get(4))} @ {fps:.0f} FPS, {total} frames")
    print(f"Signal log: {len(signal_cycles)} cycles")
    print(f"\nSPACE = pause, q = quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        t = frame_num / fps

        phase, cyc, time_left = get_phase(events, t)

        # Phase color
        if "NS" in phase and "YELLOW" not in phase:
            color = (0, 255, 0)
            label = "NS GREEN"
        elif "EW" in phase and "YELLOW" not in phase:
            color = (0, 200, 255)
            label = "EW GREEN"
        elif "YELLOW" in phase:
            color = (0, 255, 255)
            label = phase.replace("_", " ")
        else:
            color = (200, 200, 200)
            label = phase

        # Background panel
        px, py = 20, 10
        overlay = frame.copy()
        panel_h = 170 if cyc else 55
        cv2.rectangle(overlay, (px - 10, py - 5), (px + 420, py + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Traffic light icon
        cv2.circle(frame, (px + 10, py + 18), 12, color, -1)
        cv2.circle(frame, (px + 10, py + 18), 13, (255, 255, 255), 2)
        cv2.putText(frame, f"{label}  ({time_left:.1f}s left)",
                    (px + 30, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Progress bar
        if cyc:
            bar_y = py + 38
            is_ew = "EW" in phase
            total_dur = cyc["ew_green"] if is_ew else cyc["ns_green"]
            elapsed = total_dur - time_left
            progress = min(1.0, elapsed / total_dur) if total_dur > 0 else 0
            cv2.rectangle(frame, (px, bar_y), (px + 390, bar_y + 8), (60, 60, 60), -1)
            cv2.rectangle(frame, (px, bar_y), (px + int(390 * progress), bar_y + 8), color, -1)

            # Scoring details
            y = py + 58
            ns_s, ew_s = cyc["ns_score"], cyc["ew_score"]
            ns_g, ew_g = cyc["ns_green"], cyc["ew_green"]
            total_s = ns_s + ew_s if (ns_s + ew_s) > 0 else 1

            header = f"Cycle {cyc['cycle']}"
            if "vehicles" in cyc:
                header += f"  |  {cyc['vehicles']} vehicles"
            cv2.putText(frame, header,
                        (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

            # NS bar
            cv2.putText(frame, f"NS: {ns_s:.1f}", (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (px + 90, y - 12), (px + 90 + int(200 * ns_s / total_s), y), (0, 255, 0), -1)
            cv2.putText(frame, f"{ns_g:.0f}s", (px + 300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 25

            # EW bar
            cv2.putText(frame, f"EW: {ew_s:.1f}", (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.rectangle(frame, (px + 90, y - 12), (px + 90 + int(200 * ew_s / total_s), y), (0, 200, 255), -1)
            cv2.putText(frame, f"{ew_g:.0f}s", (px + 300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            y += 22

            cv2.putText(frame, f"Green split: NS={ns_g:.0f}s  EW={ew_g:.0f}s  (90s budget)",
                        (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            if "details" in cyc:
                y += 18
                cv2.putText(frame, cyc["details"], (px, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Frame info bottom
        cv2.putText(frame, f"t={t:.1f}s  frame {frame_num}/{total}",
                    (px, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Adaptive Signal Phases", frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            while True:
                k = cv2.waitKey(30) & 0xFF
                if k == ord(' ') or k == ord('q'):
                    break
            if k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
