"""
Transition Probability Tracker (Coordinate-Based)
===================================================
Usage:
    python transition_tracker.py --video control.mp4

Every cell tracks transition probabilities to its 9 neighbors
using actual cell coordinates (not direction names).

Cell (10, 20) stores:
    (9,19): prob   (10,19): prob   (11,19): prob
    (9,20): prob   (10,20): prob   (11,20): prob
    (9,21): prob   (10,21): prob   (11,21): prob

Controls:
    SPACE : Pause / Resume
    'q'   : Quit
    'f'   : Toggle flow arrows
    'g'   : Toggle grid
    'p'   : Print transition stats
"""

import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

APPROACH_MARGIN       = 2    # cells from edge = vehicle entered from that side
STOP_FRAMES_THRESHOLD = 8    # consecutive frames in same cell = stopped
CREDIT_DECAY          = 0.85 # weight decay for older trajectory cells
MIN_TRAJ_FOR_CREDIT   = 3    # minimum unique cells before assigning credit
MODEL_WARMUP          = 50   # transitions needed before credit assignment starts


# ═══════════════════════════════════════════════════════════════════════════
# GRID UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_cell(x, y, cell_size):
    return (int(x // cell_size), int(y // cell_size))

def cell_to_pixel_center(cell_x, cell_y, cell_size):
    px = cell_x * cell_size + cell_size // 2
    py = cell_y * cell_size + cell_size // 2
    return (px, py)

def get_neighbors(cell):
    cx, cy = cell
    return [
        (cx-1, cy-1), (cx, cy-1), (cx+1, cy-1),
        (cx-1, cy),               (cx+1, cy),
        (cx-1, cy+1), (cx, cy+1), (cx+1, cy+1),
    ]

def clamp_to_neighbor(from_cell, to_cell):
    dx = max(-1, min(1, to_cell[0] - from_cell[0]))
    dy = max(-1, min(1, to_cell[1] - from_cell[1]))
    return (from_cell[0] + dx, from_cell[1] + dy)

def get_approach(cell, grid_w, grid_h):
    """
    Determine which edge of the frame a vehicle entered from.
    Returns 'TOP', 'BOTTOM', 'LEFT', 'RIGHT', or 'UNKNOWN'.
    No compass directions — purely frame-geometry based.
    """
    cx, cy = cell
    m = APPROACH_MARGIN
    if cy <= m:                 return "TOP"
    if cy >= grid_h - m:       return "BOTTOM"
    if cx <= m:                 return "LEFT"
    if cx >= grid_w - m:       return "RIGHT"
    return "UNKNOWN"

def draw_grid(frame, cell_size, k=None, color=(150, 150, 150), thickness=1):
    h, w = frame.shape[:2]
    for x in range(0, w, cell_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    for y in range(0, h, cell_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)
    label_every = max(1, cell_size // 15)
    for cx in range(0, w // cell_size, label_every):
        for cy in range(0, h // cell_size, label_every):
            px = cx * cell_size + 2
            py = cy * cell_size + 12
            label = f"{cx},{cy}"
            cv2.putText(frame, label, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# EXIT ZONE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class ExitZoneClassifier:
    """
    Maps a predicted path endpoint to a named exit label.
    Names are set by the operator at setup — e.g. "ExitA", "ExitB", "ExitC".
    The algorithm doesn't know or care what they mean geometrically.

    Setup example:
        classifier = ExitZoneClassifier()
        classifier.add_zone("ExitA", [(0,5),(0,6),(0,7)])
        classifier.add_zone("ExitB", [(10,0),(11,0),(12,0)])
        classifier.add_zone("ExitC", [(19,5),(19,6),(19,7)])
    """

    def __init__(self):
        self.zones = {}   # {label: set of (cx, cy)}

    def add_zone(self, label, cells):
        self.zones[label] = set(map(tuple, cells))

    def classify(self, predicted_path):
        """Return label for the endpoint of predicted_path, or 'UNKNOWN'."""
        if not predicted_path:
            return "UNKNOWN"
        end_cell = predicted_path[-1]
        for label, zone_cells in self.zones.items():
            if end_cell in zone_cells:
                return label
        return "UNKNOWN"

    def is_configured(self):
        return len(self.zones) > 0

    def all_labels(self):
        return list(self.zones.keys())


# ═══════════════════════════════════════════════════════════════════════════
# TRANSITION MODEL
# ═══════════════════════════════════════════════════════════════════════════

class TransitionModel:
    """
    Learns P(c' | c) for each cell using actual coordinates.
    Records transitions only on movement (no self-loops in model —
    stop detection is handled separately by QueueManager).
    """

    def __init__(self, cell_size, grid_cols=None):
        self.cell_size = cell_size
        self.grid_cols = grid_cols
        self.cells = {}
        self.cell_total = defaultdict(float)
        self.endpoints = []
        self.startpoints = []
        self.total_transitions = 0
        self.total_tracks_completed = 0

    def _ensure_cell(self, cell):
        if cell not in self.cells:
            self.cells[cell] = {n: 0.0 for n in get_neighbors(cell)}

    def record_transition(self, from_cell, to_cell):
        clamped = clamp_to_neighbor(from_cell, to_cell)
        self._ensure_cell(from_cell)
        self.cells[from_cell][clamped] += 1
        self.cell_total[from_cell] += 1
        self.total_transitions += 1

    def record_endpoint(self, cell, track_id, frame_number):
        self.endpoints.append({"cell": cell, "track_id": track_id, "frame": frame_number})
        self.total_tracks_completed += 1

    def record_startpoint(self, cell, track_id, frame_number):
        self.startpoints.append({"cell": cell, "track_id": track_id, "frame": frame_number})

    def get_probabilities(self, cell):
        self._ensure_cell(cell)
        total = self.cell_total[cell]
        neighbors = get_neighbors(cell)
        n = len(neighbors)
        return {nb: (self.cells[cell][nb] + 1) / (total + n) for nb in neighbors}

    def get_counts(self, cell):
        self._ensure_cell(cell)
        return dict(self.cells[cell])

    def get_most_likely_next(self, cell):
        probs = self.get_probabilities(cell)
        if not any(probs.values()):
            return None, 0
        best = max(probs, key=probs.get)
        return best, probs[best]

    def predict_path(self, start_cell, steps=10):
        path = [start_cell]
        current = start_cell
        for _ in range(steps):
            next_cell, prob = self.get_most_likely_next(current)
            if next_cell is None or prob < 0.1:
                break
            path.append(next_cell)
            current = next_cell
        return path

    def get_flow_vector(self, cell):
        probs = self.get_probabilities(cell)
        dx, dy = 0.0, 0.0
        for (nx, ny), prob in probs.items():
            dx += (nx - cell[0]) * prob
            dy += (ny - cell[1]) * prob
        length = np.sqrt(dx**2 + dy**2)
        if length > 0.01:
            dx /= length
            dy /= length
        return (dx, dy)

    def print_cell(self, cell):
        probs  = self.get_probabilities(cell)
        counts = self.get_counts(cell)
        total  = self.cell_total[cell]
        cx, cy = cell
        eight     = get_neighbors(cell)
        neighbors = eight[:4] + [cell] + eight[4:]
        probs.setdefault(cell, 0.0)
        counts.setdefault(cell, 0.0)
        print(f"\n  Cell ({cx},{cy}) — {int(total)} total transitions:")
        print(f"  ┌────────────────────┬────────────────────┬────────────────────┐")
        for row_start in [0, 3, 6]:
            n = neighbors[row_start:row_start+3]
            print(f"  │ ({n[0][0]:>2},{n[0][1]:>2}) {probs[n[0]]:>5.1%}  │"
                  f" ({n[1][0]:>2},{n[1][1]:>2}) {probs[n[1]]:>5.1%}  │"
                  f" ({n[2][0]:>2},{n[2][1]:>2}) {probs[n[2]]:>5.1%}  │")
            print(f"  │       ({int(counts[n[0]]):>4})     │"
                  f"       ({int(counts[n[1]]):>4})     │"
                  f"       ({int(counts[n[2]]):>4})     │")
            if row_start < 6:
                print(f"  ├────────────────────┼────────────────────┼────────────────────┤")
        print(f"  └────────────────────┴────────────────────┴────────────────────┘")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  Transition Model Summary (Coordinate-Based)")
        print(f"  Total transitions : {self.total_transitions}")
        print(f"  Cells with data   : {len(self.cell_total)}")
        print(f"  Tracks completed  : {self.total_tracks_completed}")
        print(f"{'='*60}")
        sorted_cells = sorted(self.cell_total.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 busiest cells:")
        for cell, _ in sorted_cells[:5]:
            self.print_cell(cell)
        if self.endpoints:
            ep_counts = defaultdict(int)
            for ep in self.endpoints:
                ep_counts[ep["cell"]] += 1
            top_ep = sorted(ep_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  Top 5 exit areas (where vehicles vanish):")
            for cell, count in top_ep:
                print(f"    Cell {cell}: {count} vehicles")

    def save(self, filepath):
        k = self.grid_cols
        data = {
            "cell_size": self.cell_size,
            "grid_cols": k,
            "total_transitions": self.total_transitions,
            "total_tracks_completed": self.total_tracks_completed,
            "cells": {}, "endpoints": [], "startpoints": []
        }
        for cell in self.cell_total:
            key = f"{cell[0]},{cell[1]}"
            probs  = self.get_probabilities(cell)
            counts = self.get_counts(cell)
            data["cells"][key] = {
                "total_transitions": int(self.cell_total[cell]),
                "neighbors": {}
            }
            for neighbor, prob in probs.items():
                n_key = f"{neighbor[0]},{neighbor[1]}"
                data["cells"][key]["neighbors"][n_key] = {
                    "count": int(counts[neighbor]),
                    "probability": round(prob, 4)
                }
        for ep in self.endpoints:
            data["endpoints"].append({
                "cell": list(ep["cell"]),
                "track_id": ep["track_id"], "frame": ep["frame"]
            })
        for sp in self.startpoints:
            data["startpoints"].append({
                "cell": list(sp["cell"]),
                "track_id": sp["track_id"], "frame": sp["frame"]
            })
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  Model saved to: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# QUEUE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class QueueManager:
    """
    Tracks live vehicle counts per approach edge (TOP/BOTTOM/LEFT/RIGHT).

    Lifecycle per vehicle:
        1. Appears on edge         → register()
        2. Stops (N frames same cell) → credit assigned from trajectory lookback
        3. Moves again             → credit removed, re-evaluated on next stop
        4. Exits / disappears      → remove()

    Queue state (for signal optimizer):
        {
            "TOP":    {"total": 3, "intentions": {"ExitA": 1.8, "ExitB": 1.2}},
            "BOTTOM": {"total": 5, "intentions": {"ExitA": 2.1, "ExitC": 2.9}},
            "LEFT":   {"total": 1, "intentions": {"ExitB": 0.8}},
            "RIGHT":  {"total": 0, "intentions": {}}
        }
    """

    EDGES = ["TOP", "BOTTOM", "LEFT", "RIGHT"]

    def __init__(self, grid_w, grid_h, exit_classifier=None):
        self.grid_w     = grid_w
        self.grid_h     = grid_h
        self.classifier = exit_classifier or ExitZoneClassifier()

        self.queue           = defaultdict(int)
        self.intention_queue = defaultdict(lambda: defaultdict(float))

        self.tid_to_approach  = {}
        self.tid_to_credit    = {}
        self.tid_to_status    = {}   # "moving" | "stopped" | "confirmed"

        self.consecutive_same = defaultdict(int)
        self.credit_assigned  = set()

        self.total_registered = 0
        self.total_confirmed  = 0
        self.total_removed    = 0

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, tid, first_cell):
        """
        Call once on first detection.
        Only registers if vehicle appeared on a real frame edge.
        Returns approach string or None.
        """
        approach = get_approach(first_cell, self.grid_w, self.grid_h)
        if approach == "UNKNOWN":
            return None
        self.tid_to_approach[tid] = approach
        self.tid_to_credit[tid]   = {}
        self.tid_to_status[tid]   = "moving"
        self.queue[approach]     += 1
        self.total_registered    += 1
        return approach

    # ── Per-frame update ──────────────────────────────────────────────────

    def update(self, tid, current_cell, prev_cell, trajectory_history, trans):
        """
        Call every frame for every tracked vehicle.
        Handles stop detection and triggers credit assignment automatically.
        """
        if tid not in self.tid_to_approach:
            return

        # Stop detection
        if prev_cell is not None and current_cell == prev_cell:
            self.consecutive_same[tid] += 1
        else:
            self.consecutive_same[tid] = 0
            # Vehicle moved again after stop → remove old credit, reset
            if tid in self.credit_assigned:
                self.credit_assigned.discard(tid)
                self._remove_credit(tid)
                self.tid_to_status[tid] = "moving"

        just_stopped = (
            self.consecutive_same[tid] == STOP_FRAMES_THRESHOLD
            and tid not in self.credit_assigned
            and trans.total_transitions >= MODEL_WARMUP
        )

        if just_stopped:
            credit = self._compute_credit(tid, trajectory_history, trans)
            if credit:
                self._apply_credit(tid, credit)
                self.credit_assigned.add(tid)
                self.tid_to_status[tid] = "stopped"

    # ── Credit computation (trajectory lookback) ──────────────────────────

    def _compute_credit(self, tid, trajectory_history, trans):
        """
        Look back at the vehicle's partial trajectory.
        Weight each cell's predicted exit by recency (most recent = highest weight).
        Normalize to produce a probability distribution over exit zones.

        Returns: {"ExitA": 0.6, "ExitB": 0.3, "ExitC": 0.1} or {}
        """
        traj = trajectory_history.get(tid, [])
        if len(traj) < MIN_TRAJ_FOR_CREDIT:
            return {}
        if not self.classifier.is_configured():
            return {}

        # Unique cells in order
        unique_cells = []
        for entry in traj:
            cell = (entry[3], entry[4])
            if not unique_cells or unique_cells[-1] != cell:
                unique_cells.append(cell)

        # Use last 10 unique cells, most recent gets highest weight
        recent = unique_cells[-10:]
        credit = defaultdict(float)
        weight = 1.0

        for cell in reversed(recent):
            path  = trans.predict_path(cell, steps=20)
            label = self.classifier.classify(path)
            if label != "UNKNOWN":
                credit[label] += weight
            weight *= CREDIT_DECAY

        if not credit:
            return {}

        total = sum(credit.values())
        return {k: round(v / total, 4) for k, v in credit.items()}

    def _apply_credit(self, tid, credit):
        approach = self.tid_to_approach[tid]
        self.tid_to_credit[tid] = credit
        for label, prob in credit.items():
            self.intention_queue[approach][label] += prob

    def _remove_credit(self, tid):
        approach   = self.tid_to_approach.get(tid)
        old_credit = self.tid_to_credit.get(tid, {})
        if approach:
            for label, prob in old_credit.items():
                self.intention_queue[approach][label] = max(
                    0.0, self.intention_queue[approach][label] - prob
                )
        self.tid_to_credit[tid] = {}

    # ── Exit confirmation ─────────────────────────────────────────────────

    def confirm_exit(self, tid, actual_label):
        """
        Call when vehicle reaches a known exit zone.
        Replaces soft fractional credit with a hard confirmed +1.
        """
        if tid not in self.tid_to_approach:
            return
        approach = self.tid_to_approach[tid]
        self._remove_credit(tid)
        if actual_label != "UNKNOWN":
            self.intention_queue[approach][actual_label] += 1.0
        self.tid_to_status[tid] = "confirmed"
        self.total_confirmed += 1
        self._cleanup(tid)

    # ── Removal (disappeared / tracker loss) ─────────────────────────────

    def remove(self, tid):
        """Call when vehicle disappears without confirmed exit."""
        if tid not in self.tid_to_approach:
            return
        approach = self.tid_to_approach[tid]
        self._remove_credit(tid)
        self.queue[approach] = max(0, self.queue[approach] - 1)
        self.total_removed += 1
        self._cleanup(tid)

    def _cleanup(self, tid):
        self.tid_to_approach.pop(tid, None)
        self.tid_to_credit.pop(tid, None)
        self.tid_to_status.pop(tid, None)
        self.consecutive_same.pop(tid, None)
        self.credit_assigned.discard(tid)

    # ── Signal optimizer interface ────────────────────────────────────────

    def get_queue_state(self):
        """
        Returns full queue state dict for signal optimizer.
        Approach names are TOP/BOTTOM/LEFT/RIGHT — no compass assumptions.
        """
        state = {}
        for edge in self.EDGES:
            state[edge] = {
                "total":      self.queue.get(edge, 0),
                "intentions": dict(self.intention_queue.get(edge, {}))
            }
        return state

    def get_pressure(self):
        """Simple pressure score — just total queue per approach."""
        return {edge: self.queue.get(edge, 0) for edge in self.EDGES}

    # ── Visualization ─────────────────────────────────────────────────────

    def draw_hud(self, frame):
        """Draw live queue panel top-right corner."""
        h, w    = frame.shape[:2]
        panel_x = w - 310
        panel_y = 10
        panel_h = 20 + len(self.EDGES) * 22 + 10

        cv2.rectangle(frame, (panel_x - 5, panel_y),
                      (w - 5, panel_y + panel_h), (0, 0, 0), -1)
        cv2.putText(frame, "QUEUE STATE", (panel_x, panel_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        y = panel_y + 34
        for edge in self.EDGES:
            total      = self.queue.get(edge, 0)
            intentions = self.intention_queue.get(edge, {})
            color      = (0, 255, 0) if total > 0 else (80, 80, 80)

            # e.g.  TOP:  3   ExitA:1.8  ExitB:1.2
            intent_str = "  ".join(
                f"{k}:{v:.1f}" for k, v in sorted(intentions.items()) if v > 0.05
            ) or "—"

            cv2.putText(frame, f"{edge:<7}{total:>2}  {intent_str}",
                        (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
            y += 22

        return frame

    def draw_vehicle_label(self, frame, tid, x1, y1, y2):
        """Draw per-vehicle label: approach edge + status + credit distribution."""
        approach = self.tid_to_approach.get(tid, "?")
        status   = self.tid_to_status.get(tid, "?")
        credit   = self.tid_to_credit.get(tid, {})

        color = {
            "moving":    (0, 255, 255),
            "stopped":   (0, 165, 255),
            "confirmed": (0, 255, 0),
        }.get(status, (200, 200, 200))

        cv2.putText(frame, f"ID:{tid} [{approach}] {status}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        if credit:
            credit_str = "  ".join(f"{k}:{v:.0%}" for k, v in credit.items())
            cv2.putText(frame, credit_str,
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1)
        return frame

    def print_status(self):
        print(f"\n  ── Queue Status {'─'*40}")
        for edge in self.EDGES:
            total   = self.queue.get(edge, 0)
            intents = self.intention_queue.get(edge, {})
            istr    = "  ".join(f"{k}:{v:.1f}" for k, v in sorted(intents.items()) if v > 0.05) or "no data"
            print(f"    {edge:<7}: {total:>3} vehicles  │  {istr}")
        print(f"  Registered:{self.total_registered}  "
              f"Confirmed:{self.total_confirmed}  "
              f"Removed:{self.total_removed}")
        print(f"  {'─'*55}")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def draw_flow_arrows(frame, model, cell_size, min_transitions=3):
    if not model.cell_total:
        return frame
    max_count = max(model.cell_total.values())
    for cell, count in model.cell_total.items():
        if count < min_transitions:
            continue
        dx, dy = model.get_flow_vector(cell)
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            continue
        cx, cy    = cell_to_pixel_center(cell[0], cell[1], cell_size)
        arrow_len = cell_size * 0.4
        ex        = int(cx + dx * arrow_len)
        ey        = int(cy + dy * arrow_len)
        intensity = min(count / max_count, 1.0)
        color     = (int(255*(1-intensity)), int(255*(1-intensity)), 255)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2, tipLength=0.4)
    return frame

def draw_endpoints(frame, model, cell_size):
    ep_counts = defaultdict(int)
    for ep in model.endpoints:
        ep_counts[ep["cell"]] += 1
    if not ep_counts:
        return frame
    max_count = max(ep_counts.values())
    for cell, count in ep_counts.items():
        cx, cy = cell_to_pixel_center(cell[0], cell[1], cell_size)
        radius = int(3 + 10 * (count / max_count))
        cv2.circle(frame, (cx, cy), radius, (0, 255, 255), -1)
    return frame

def draw_predicted_paths(frame, model, active_cells, cell_size):
    for cell in active_cells:
        path = model.predict_path(cell, steps=8)
        if len(path) < 2:
            continue
        for i in range(len(path) - 1):
            p1    = cell_to_pixel_center(path[i][0],   path[i][1],   cell_size)
            p2    = cell_to_pixel_center(path[i+1][0], path[i+1][1], cell_size)
            alpha = 1.0 - (i / len(path))
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.line(frame, p1, p2, color, 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════

def save_all(trans, trajectory_history, grid_w, cell_size, video_path):
    stem = Path(video_path).stem

    out_trans = Path("data/transitions")
    out_trans.mkdir(parents=True, exist_ok=True)
    trans.save(str(out_trans / f"{stem}_transitions.json"))

    trajectories_data = {}
    for tid, traj in trajectory_history.items():
        cells_seq = []
        for entry in traj:
            cell = (entry[3], entry[4])
            if not cells_seq or cells_seq[-1] != cell:
                cells_seq.append(cell)
        if len(cells_seq) >= 3:
            trajectories_data[str(tid)] = {
                "cells":  [list(c) for c in cells_seq],
                "start":  list(cells_seq[0]),
                "end":    list(cells_seq[-1]),
                "length": len(cells_seq)
            }

    out_traj  = Path("data/trajectories")
    out_traj.mkdir(parents=True, exist_ok=True)
    traj_path = out_traj / f"{stem}_trajectories.json"
    with open(str(traj_path), 'w') as f:
        json.dump({
            "cell_size":    cell_size,
            "grid_cols":    grid_w,
            "total_tracks": len(trajectories_data),
            "trajectories": trajectories_data
        }, f, indent=2)
    print(f"  Trajectories saved : {traj_path}")
    print(f"  Usable trajectories: {len(trajectories_data)}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Coordinate-Based Transition Tracker")
    parser.add_argument("--video",      required=True,               help="Path to video file")
    parser.add_argument("--model_path", default="../models/best.pt", help="YOLO model path")
    parser.add_argument("--cell_size",  type=int, default=30,        help="Grid cell size in pixels")
    args = parser.parse_args()

    cell_size = args.cell_size
    yolo      = YOLO(args.model_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.video}'")
        return

    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    grid_w       = frame_w // cell_size
    grid_h       = frame_h // cell_size

    print(f"Video : {frame_w}x{frame_h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Grid  : {cell_size}px → {grid_w} x {grid_h} = {grid_w*grid_h} cells")

    trans     = TransitionModel(cell_size, grid_cols=grid_w)

    # ── Define exit zones here (operator sets names and cell lists) ────────
    exit_zones = ExitZoneClassifier()
    # exit_zones.add_zone("ExitA", [(0,5),(0,6),(0,7)])
    # exit_zones.add_zone("ExitB", [(10,0),(11,0),(12,0)])
    # exit_zones.add_zone("ExitC", [(19,5),(19,6),(19,7)])
    # ──────────────────────────────────────────────────────────────────────

    queue_mgr = QueueManager(grid_w, grid_h, exit_classifier=exit_zones)

    trajectory_history = defaultdict(list)
    prev_cells         = {}
    prev_frame_ids     = set()
    started_ids        = set()
    frame_number       = 0

    show_flow        = True
    show_grid        = True
    show_predictions = True
    paused           = False

    print(f"\nRunning... SPACE pause | 'q' quit | 'f' flow | 'g' grid | 'p' stats")
    print(f"  LEFT/RIGHT arrows: step frames while paused\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number     += 1
        current_frame_ids = set()
        active_cells      = []

        if show_grid:
            frame = draw_grid(frame, cell_size, k=grid_w)

        results = yolo.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_cell = pixel_to_cell(cx, cy, cell_size)
                current_frame_ids.add(tid)
                active_cells.append(current_cell)

                trajectory_history[tid].append(
                    (cx, cy, frame_number, current_cell[0], current_cell[1])
                )

                # ── First detection ───────────────────────────────────────
                if tid not in started_ids:
                    queue_mgr.register(tid, current_cell)
                    trans.record_startpoint(current_cell, tid, frame_number)
                    started_ids.add(tid)

                # ── Record movement transition ────────────────────────────
                if tid in prev_cells and prev_cells[tid] != current_cell:
                    trans.record_transition(prev_cells[tid], current_cell)

                # ── Queue update (stop detection + credit) ────────────────
                queue_mgr.update(
                    tid, current_cell, prev_cells.get(tid),
                    trajectory_history, trans
                )

                prev_cells[tid] = current_cell

                # ── Check if vehicle reached an exit zone ─────────────────
                if exit_zones.is_configured():
                    path  = trans.predict_path(current_cell, steps=1)
                    label = exit_zones.classify([current_cell])
                    if label != "UNKNOWN":
                        queue_mgr.confirm_exit(tid, label)

                # ── Draw bounding box ─────────────────────────────────────
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                frame = queue_mgr.draw_vehicle_label(frame, tid, x1, y1, y2)

                # ── Show most likely next cell ────────────────────────────
                next_cell, prob = trans.get_most_likely_next(current_cell)
                if next_cell and prob > 0.3:
                    cv2.putText(frame, f"next:{next_cell[0]},{next_cell[1]} {prob:.0%}",
                                (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.38, (255, 255, 0), 1)

        # ── Disappeared vehicles ──────────────────────────────────────────
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            queue_mgr.remove(tid)
            if tid in trajectory_history and len(trajectory_history[tid]) > 5:
                last = trajectory_history[tid][-1]
                trans.record_endpoint((last[3], last[4]), tid, frame_number)
                if trans.total_tracks_completed % 10 == 0:
                    print(f"  [Frame {frame_number}] "
                          f"Completed:{trans.total_tracks_completed} | "
                          f"Transitions:{trans.total_transitions}")

        prev_frame_ids = current_frame_ids

        # ── Visualizations ────────────────────────────────────────────────
        if show_flow and trans.total_transitions > 0:
            frame = draw_flow_arrows(frame, trans, cell_size, min_transitions=3)
            frame = draw_endpoints(frame, trans, cell_size)

        if show_predictions and trans.total_transitions > 100:
            frame = draw_predicted_paths(frame, trans, active_cells, cell_size)

        frame = queue_mgr.draw_hud(frame)

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(frame,
                    f"Frame:{frame_number}/{total_frames} | "
                    f"Trans:{trans.total_transitions} | "
                    f"Done:{trans.total_tracks_completed}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Cells:{len(trans.cell_total)} | "
                    f"Flow:{'ON' if show_flow else 'OFF'}(f) | "
                    f"Grid:{'ON' if show_grid else 'OFF'}(g) | "
                    f"{'PAUSED' if paused else 'PLAYING'}(space)",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

        cv2.imshow("Transition Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'RESUMED'}")
            while paused:
                key2 = cv2.waitKey(30) & 0xFF
                if key2 == ord(' '):
                    paused = False
                elif key2 == ord('q'):
                    paused = False
                    cap.release()
                    cv2.destroyAllWindows()
                    trans.summary()
                    queue_mgr.print_status()
                    save_all(trans, trajectory_history, grid_w, cell_size, args.video)
                    return
                elif key2 in (81, 2):   # LEFT arrow
                    new_pos = max(0, frame_number - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_number = new_pos
                    ret2, f2 = cap.read()
                    if ret2:
                        frame_number += 1
                        if show_grid: f2 = draw_grid(f2, cell_size, k=grid_w)
                        f2 = queue_mgr.draw_hud(f2)
                        cv2.putText(f2, f"Frame:{frame_number} | PAUSED",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255,255,255), 2)
                        cv2.imshow("Transition Tracker", f2)
                    print(f"  ← {frame_number}")
                elif key2 in (83, 3):   # RIGHT arrow
                    ret2, f2 = cap.read()
                    if ret2:
                        frame_number += 1
                        if show_grid: f2 = draw_grid(f2, cell_size, k=grid_w)
                        f2 = queue_mgr.draw_hud(f2)
                        cv2.putText(f2, f"Frame:{frame_number} | PAUSED",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255,255,255), 2)
                        cv2.imshow("Transition Tracker", f2)
                    print(f"  → {frame_number}")
                elif key2 == ord('f'): show_flow = not show_flow
                elif key2 == ord('g'): show_grid = not show_grid
                elif key2 == ord('p'):
                    trans.summary()
                    queue_mgr.print_status()
        elif key == ord('f'): show_flow = not show_flow
        elif key == ord('g'): show_grid = not show_grid
        elif key == ord('p'):
            trans.summary()
            queue_mgr.print_status()

    # ── Final cleanup ─────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    for tid in current_frame_ids:
        if tid in trajectory_history and len(trajectory_history[tid]) > 5:
            last = trajectory_history[tid][-1]
            trans.record_endpoint((last[3], last[4]), tid, frame_number)

    trans.summary()
    queue_mgr.print_status()
    save_all(trans, trajectory_history, grid_w, cell_size, args.video)

    print(f"\n  Next steps:")
    print(f"    1. Define exit zones → credit assignment activates")
    print(f"    2. Queue state feeds signal optimizer")
    print(f"    3. confirm_exit() closes the loop with ground truth\n")


if __name__ == "__main__":
    main()
