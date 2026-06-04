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
from grid_utils import cell_to_id, id_to_cell

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent



# ═══════════════════════════════════════════════════════════════════════════
# GRID UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_cell(x, y, cell_size):
    """Convert pixel coordinates to grid cell."""
    return (int(x // cell_size), int(y // cell_size))


def cell_to_pixel_center(cell_x, cell_y, cell_size):
    """Convert grid cell back to pixel center."""
    px = cell_x * cell_size + cell_size // 2
    py = cell_y * cell_size + cell_size // 2
    return (px, py)


def get_neighbors(cell):
    """
    Get the 8 neighboring cells (no stay/self).
    Returns list of (cx, cy) coordinates.
    """
    cx, cy = cell
    return [
        (cx-1, cy-1), (cx, cy-1), (cx+1, cy-1),   # top row
        (cx-1, cy),              (cx+1, cy),        # middle row (no self)
        (cx-1, cy+1), (cx, cy+1), (cx+1, cy+1),    # bottom row
    ]


def clamp_to_neighbor(from_cell, to_cell):
    """
    If a car jumps more than 1 cell, clamp to the nearest neighbor.
    Keeps everything within the 9-cell neighborhood.
    """
    dx = to_cell[0] - from_cell[0]
    dy = to_cell[1] - from_cell[1]
    dx = max(-1, min(1, dx))
    dy = max(-1, min(1, dy))
    return (from_cell[0] + dx, from_cell[1] + dy)


def draw_grid(frame, cell_size, k=None, color=(150, 150, 150), thickness=1):
    """Draw grid overlay with linear cell ID labels (or col,row if k is None)."""
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
            # Use linear ID when k is available, fall back to (col,row)
            label = str(cell_to_id(cx, cy, k)) if k is not None else f"{cx},{cy}"
            cv2.putText(frame, label, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# TRANSITION MODEL (COORDINATE-BASED)
# ═══════════════════════════════════════════════════════════════════════════

class TransitionModel:
    """
    Learns P(c' | c) for each cell using actual coordinates.

    For cell (10, 20), stores counts and probabilities for all 9 neighbors:
        (9,19)  (10,19)  (11,19)
        (9,20)  (10,20)  (11,20)
        (9,21)  (10,21)  (11,21)
    """

    def __init__(self, cell_size, grid_cols=None):
        self.cell_size = cell_size
        self.grid_cols = grid_cols  # k = cells per row, used for linear ID encoding

        # {(cx,cy): {(nx,ny): count, ...}} — 9 entries per cell
        self.cells = {}

        # Total transitions per cell
        self.cell_total = defaultdict(float)

        # Endpoints and startpoints
        self.endpoints = []
        self.startpoints = []

        # Stats
        self.total_transitions = 0
        self.total_tracks_completed = 0

    def _ensure_cell(self, cell):
        """Make sure a cell has all 9 neighbor slots initialized."""
        if cell not in self.cells:
            self.cells[cell] = {n: 0.0 for n in get_neighbors(cell)}

    def record_transition(self, from_cell, to_cell):
        """
        Record one cell-to-cell transition.
        Clamps to nearest neighbor if car jumped multiple cells.
        """
        clamped = clamp_to_neighbor(from_cell, to_cell)
        self._ensure_cell(from_cell)
        self.cells[from_cell][clamped] += 1
        self.cell_total[from_cell] += 1
        self.total_transitions += 1

    def record_endpoint(self, cell, track_id, frame_number):
        """Record where a vehicle disappeared."""
        self.endpoints.append({
            "cell": cell,
            "track_id": track_id,
            "frame": frame_number
        })
        self.total_tracks_completed += 1

    def record_startpoint(self, cell, track_id, frame_number):
        """Record where a vehicle first appeared."""
        self.startpoints.append({
            "cell": cell,
            "track_id": track_id,
            "frame": frame_number
        })

    def get_probabilities(self, cell):
        """
        Get P(c' | c) for all 8 neighbors with Laplace smoothing (epsilon=1).

        Laplace smoothing adds 1 to every neighbor count so that no
        transition probability is ever exactly 0.  The smoothed estimate is:

            P(c' | c) = (count(c→c') + 1) / (total_transitions + num_neighbors)

        Returns:
            dict: {(nx, ny): probability, ...} — 8 entries, sums to 1.0
        """
        self._ensure_cell(cell)
        total = self.cell_total[cell]
        neighbors = get_neighbors(cell)
        n = len(neighbors)  # always 8
        # Laplace smoothing: uniform prior prevents zero probabilities
        return {nb: (self.cells[cell][nb] + 1) / (total + n) for nb in neighbors}

    def get_counts(self, cell):
        """Get raw counts for all 9 neighbors."""
        self._ensure_cell(cell)
        return dict(self.cells[cell])

    def get_most_likely_next(self, cell):
        """
        Get the most likely next cell coordinate.

        Returns:
            ((nx, ny), probability) or (None, 0)
        """
        probs = self.get_probabilities(cell)
        if not any(probs.values()):
            return None, 0

        best = max(probs, key=probs.get)
        return best, probs[best]

    def predict_path(self, start_cell, steps=10):
        """
        Predict future path by following most likely transitions.
        
        Args:
            start_cell: (cx, cy) starting cell
            steps: how many steps to predict
        
        Returns:
            list of (cx, cy) predicted cells
        """
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
        """Get weighted average flow direction as (dx, dy) for drawing arrows."""
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
        """Pretty-print one cell's 3x3 neighbor table with coordinates."""
        probs = self.get_probabilities(cell)
        counts = self.get_counts(cell)
        total = self.cell_total[cell]
        cx, cy = cell

        # get_neighbors returns 8 entries (no self); insert cell at center for 3×3 display
        eight = get_neighbors(cell)
        neighbors = eight[:4] + [cell] + eight[4:]
        probs.setdefault(cell, 0.0)
        counts.setdefault(cell, 0.0)

        print(f"\n  Cell ({cx},{cy}) — {int(total)} total transitions:")
        print(f"  ┌────────────────────┬────────────────────┬────────────────────┐")
        # Top row
        n = neighbors[0:3]
        print(f"  │ ({n[0][0]:>2},{n[0][1]:>2}) {probs[n[0]]:>5.1%}  │ ({n[1][0]:>2},{n[1][1]:>2}) {probs[n[1]]:>5.1%}  │ ({n[2][0]:>2},{n[2][1]:>2}) {probs[n[2]]:>5.1%}  │")
        print(f"  │       ({int(counts[n[0]]):>4})     │       ({int(counts[n[1]]):>4})     │       ({int(counts[n[2]]):>4})     │")
        print(f"  ├────────────────────┼────────────────────┼────────────────────┤")
        # Middle row
        n = neighbors[3:6]
        mid_label = " ← STAY" if n[1] == cell else ""
        print(f"  │ ({n[0][0]:>2},{n[0][1]:>2}) {probs[n[0]]:>5.1%}  │ ({n[1][0]:>2},{n[1][1]:>2}) {probs[n[1]]:>5.1%}  │ ({n[2][0]:>2},{n[2][1]:>2}) {probs[n[2]]:>5.1%}  │")
        print(f"  │       ({int(counts[n[0]]):>4})     │       ({int(counts[n[1]]):>4})     │       ({int(counts[n[2]]):>4})     │")
        print(f"  ├────────────────────┼────────────────────┼────────────────────┤")
        # Bottom row
        n = neighbors[6:9]
        print(f"  │ ({n[0][0]:>2},{n[0][1]:>2}) {probs[n[0]]:>5.1%}  │ ({n[1][0]:>2},{n[1][1]:>2}) {probs[n[1]]:>5.1%}  │ ({n[2][0]:>2},{n[2][1]:>2}) {probs[n[2]]:>5.1%}  │")
        print(f"  │       ({int(counts[n[0]]):>4})     │       ({int(counts[n[1]]):>4})     │       ({int(counts[n[2]]):>4})     │")
        print(f"  └────────────────────┴────────────────────┴────────────────────┘")

    def summary(self):
        """Print full model summary."""
        print(f"\n{'='*60}")
        print(f"  Transition Model Summary (Coordinate-Based)")
        print(f"  Total transitions: {self.total_transitions}")
        print(f"  Cells with data: {len(self.cell_total)}")
        print(f"  Tracks completed: {self.total_tracks_completed}")
        print(f"{'='*60}")

        sorted_cells = sorted(self.cell_total.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 busiest cells:")
        for cell, count in sorted_cells[:5]:
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
        """Save model to JSON using linear cell IDs as keys (instead of col,row strings)."""
        k = self.grid_cols  # cells per row — needed for linear ID computation

        data = {
            "cell_size": self.cell_size,
            "grid_cols": k,          # store k so markov_model.py can reconstruct (col,row)
            "total_transitions": self.total_transitions,
            "total_tracks_completed": self.total_tracks_completed,
            "cells": {},
            "endpoints": [],
            "startpoints": []
        }

        for cell in self.cell_total:
            # Linear ID key: id = (col+1) + k*(row)  [via cell_to_id]
            key = str(cell_to_id(cell[0], cell[1], k)) if k else f"{cell[0]},{cell[1]}"
            probs = self.get_probabilities(cell)
            counts = self.get_counts(cell)

            data["cells"][key] = {
                "total_transitions": int(self.cell_total[cell]),
                "neighbors": {}
            }

            for neighbor, prob in probs.items():
                n_key = str(cell_to_id(neighbor[0], neighbor[1], k)) if k else f"{neighbor[0]},{neighbor[1]}"
                data["cells"][key]["neighbors"][n_key] = {
                    "count": int(counts[neighbor]),
                    "probability": round(prob, 4)
                }

        for ep in self.endpoints:
            data["endpoints"].append({
                "cell": cell_to_id(ep["cell"][0], ep["cell"][1], k) if k else list(ep["cell"]),
                "track_id": ep["track_id"],
                "frame": ep["frame"]
            })

        for sp in self.startpoints:
            data["startpoints"].append({
                "cell": cell_to_id(sp["cell"][0], sp["cell"][1], k) if k else list(sp["cell"]),
                "track_id": sp["track_id"],
                "frame": sp["frame"]
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  Model saved to: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def draw_flow_arrows(frame, model, cell_size, min_transitions=3):
    """Draw arrows showing dominant flow direction per cell."""
    if not model.cell_total:
        return frame

    max_count = max(model.cell_total.values())

    for cell, count in model.cell_total.items():
        if count < min_transitions:
            continue

        dx, dy = model.get_flow_vector(cell)
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            continue

        cx, cy = cell_to_pixel_center(cell[0], cell[1], cell_size)
        arrow_len = cell_size * 0.4
        ex = int(cx + dx * arrow_len)
        ey = int(cy + dy * arrow_len)

        intensity = min(count / max_count, 1.0)
        color = (
            int(255 * (1 - intensity)),
            int(255 * (1 - intensity)),
            255
        )

        cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2, tipLength=0.4)

    return frame


def draw_endpoints(frame, model, cell_size):
    """Draw endpoint clusters as yellow dots."""
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
    """Draw predicted future paths for currently tracked vehicles."""
    for cell in active_cells:
        path = model.predict_path(cell, steps=8)
        if len(path) < 2:
            continue

        for i in range(len(path) - 1):
            p1 = cell_to_pixel_center(path[i][0], path[i][1], cell_size)
            p2 = cell_to_pixel_center(path[i+1][0], path[i+1][1], cell_size)
            # Fade color along prediction
            alpha = 1.0 - (i / len(path))
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.line(frame, p1, p2, color, 2)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Coordinate-Based Transition Tracker")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model_path", default="../models/best.pt", help="YOLO model path")
    parser.add_argument("--cell_size", type=int, default=30, help="Grid cell size in pixels")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    args = parser.parse_args()

    cell_size = args.cell_size

    yolo = YOLO(args.model_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.video}'")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    grid_w, grid_h = frame_w // cell_size, frame_h // cell_size
    # Now that we know the grid dimensions, create TransitionModel with k=grid_w
    trans = TransitionModel(cell_size, grid_cols=grid_w)
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Grid: {cell_size}px cells → {grid_w} x {grid_h} = {grid_w * grid_h} total cells")
    print(f"Linear cell IDs: k={grid_w} cells/row (id = col + {grid_w}*(row-1), 1-based)")

    # Tracking state
    trajectory_history = defaultdict(list)
    prev_cells = {}
    prev_frame_ids = set()
    started_ids = set()
    frame_number = 0

    # Display toggles
    show_flow = True
    show_grid = True
    show_predictions = True
    paused = False

    print(f"\nRunning... SPACE pause | 'q' quit | 'f' flow | 'g' grid | 'p' stats")
    print(f"  LEFT arrow: back 1 frame | RIGHT arrow: forward 1 frame (while paused)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_frame_ids = set()
        active_cells = []

        if show_grid:
            frame = draw_grid(frame, cell_size, k=grid_w)

        # Run YOLO tracking
        results = yolo.track(frame, persist=True, classes=[3, 4, 5, 8], imgsz=args.imgsz)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_cell = pixel_to_cell(cx, cy, cell_size)
                current_frame_ids.add(tid)
                active_cells.append(current_cell)

                trajectory_history[tid].append((cx, cy, frame_number, current_cell[0], current_cell[1]))

                if tid not in started_ids:
                    trans.record_startpoint(current_cell, tid, frame_number)
                    started_ids.add(tid)

                # Record transition ONLY if car moved to a different cell
                if tid in prev_cells:
                    if prev_cells[tid] != current_cell:
                        trans.record_transition(prev_cells[tid], current_cell)

                prev_cells[tid] = current_cell

                # Draw tracking info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cell_id = cell_to_id(current_cell[0], current_cell[1], grid_w)
                cv2.putText(frame, f"ID:{tid} cell:{cell_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Show most likely next cell as linear ID
                next_cell, prob = trans.get_most_likely_next(current_cell)
                if next_cell and prob > 0.3:
                    next_id = cell_to_id(next_cell[0], next_cell[1], grid_w)
                    cv2.putText(frame, f"next:{next_id} {prob:.0%}",
                                (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Detect disappeared tracks
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            if tid in trajectory_history and len(trajectory_history[tid]) > 5:
                last = trajectory_history[tid][-1]
                trans.record_endpoint((last[3], last[4]), tid, frame_number)

                if trans.total_tracks_completed % 10 == 0:
                    print(f"  [Frame {frame_number}] Completed: {trans.total_tracks_completed} | "
                          f"Transitions: {trans.total_transitions}")

        prev_frame_ids = current_frame_ids

        # Draw visualizations
        if show_flow and trans.total_transitions > 0:
            frame = draw_flow_arrows(frame, trans, cell_size, min_transitions=3)
            frame = draw_endpoints(frame, trans, cell_size)

        if show_predictions and trans.total_transitions > 100:
            frame = draw_predicted_paths(frame, trans, active_cells, cell_size)

        # HUD
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames} | "
                    f"Transitions: {trans.total_transitions} | "
                    f"Completed: {trans.total_tracks_completed}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Cells: {len(trans.cell_total)} | "
                    f"Flow: {'ON' if show_flow else 'OFF'} (f) | "
                    f"Grid: {'ON' if show_grid else 'OFF'} (g) | "
                    f"{'PAUSED' if paused else 'PLAYING'} (space) | "
                    f"←→ step",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Coordinate-Based Transition Tracker", frame)

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
                    print(f"  RESUMED")
                    break
                elif key2 == ord('q'):
                    paused = False
                    cap.release()
                    cv2.destroyAllWindows()
                    trans.summary()
                    out_dir_trans = PROJECT_ROOT / "data" / "transitions"
                    out_dir_trans.mkdir(parents=True, exist_ok=True)
                    output_path = out_dir_trans / (Path(args.video).stem + "_transitions.json")
                    trans.save(str(output_path))
                    return
                elif key2 == 81 or key2 == 2:  # LEFT arrow
                    # Go back 2 frames (1 to undo current, 1 to go back)
                    new_pos = max(0, frame_number - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_number = new_pos
                    ret_step, frame_step = cap.read()
                    if ret_step:
                        frame_number += 1
                        if show_grid:
                            frame_step = draw_grid(frame_step, cell_size, k=grid_w)
                        if show_flow and trans.total_transitions > 0:
                            frame_step = draw_flow_arrows(frame_step, trans, cell_size)
                            frame_step = draw_endpoints(frame_step, trans, cell_size)
                        cv2.putText(frame_step, f"Frame: {frame_number}/{total_frames} | PAUSED | ← →  step",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow("Coordinate-Based Transition Tracker", frame_step)
                    print(f"  ← Frame {frame_number}")
                elif key2 == 83 or key2 == 3:  # RIGHT arrow
                    ret_step, frame_step = cap.read()
                    if ret_step:
                        frame_number += 1
                        if show_grid:
                            frame_step = draw_grid(frame_step, cell_size, k=grid_w)
                        if show_flow and trans.total_transitions > 0:
                            frame_step = draw_flow_arrows(frame_step, trans, cell_size)
                            frame_step = draw_endpoints(frame_step, trans, cell_size)
                        cv2.putText(frame_step, f"Frame: {frame_number}/{total_frames} | PAUSED | ← → step",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow("Coordinate-Based Transition Tracker", frame_step)
                    print(f"  → Frame {frame_number}")
                elif key2 == ord('f'):
                    show_flow = not show_flow
                elif key2 == ord('g'):
                    show_grid = not show_grid
                elif key2 == ord('p'):
                    trans.summary()
        elif key == ord('f'):
            show_flow = not show_flow
        elif key == ord('g'):
            show_grid = not show_grid
        elif key == ord('p'):
            trans.summary()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    for tid in current_frame_ids:
        if tid in trajectory_history and len(trajectory_history[tid]) > 5:
            last = trajectory_history[tid][-1]
            trans.record_endpoint((last[3], last[4]), tid, frame_number)

    trans.summary()

    out_dir_trans = PROJECT_ROOT / "data" / "transitions"
    out_dir_trans.mkdir(parents=True, exist_ok=True)
    output_path = out_dir_trans / (Path(args.video).stem + "_transitions.json")
    trans.save(str(output_path))

    # ── Save raw trajectories for ML training ─────────────────────────────
    trajectories_data = {}
    for tid, traj in trajectory_history.items():
        # Extract unique cells in order (remove consecutive duplicates)
        cells_seq = []
        for entry in traj:
            cell = (entry[3], entry[4])
            if not cells_seq or cells_seq[-1] != cell:
                cells_seq.append(cell)

        if len(cells_seq) >= 3:  # Only keep meaningful trajectories
            trajectories_data[str(tid)] = {
                # Store cells as linear IDs (single integers) instead of [col,row] pairs
                "cells": [cell_to_id(c[0], c[1], grid_w) for c in cells_seq],
                "start": cell_to_id(cells_seq[0][0], cells_seq[0][1], grid_w),
                "end": cell_to_id(cells_seq[-1][0], cells_seq[-1][1], grid_w),
                "length": len(cells_seq)
            }

    out_dir_traj = PROJECT_ROOT / "data" / "trajectories"
    out_dir_traj.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir_traj / (Path(args.video).stem + "_trajectories.json")
    with open(str(traj_path), 'w') as f:
        json.dump({
            "cell_size": cell_size,
            "grid_cols": grid_w,   # k stored so lstm_predictor.py can reconstruct (col,row)
            "total_tracks": len(trajectories_data),
            "trajectories": trajectories_data
        }, f, indent=2)
    print(f"  Raw trajectories saved to: {traj_path}")
    print(f"  Total usable trajectories: {len(trajectories_data)}")

    print(f"\n  Next steps:")
    print(f"    1. Cluster endpoints → auto-discover exit zones")
    print(f"    2. Train ML model on transition sequences")
    print(f"    3. Predict vehicle intention early in trajectory\n")


if __name__ == "__main__":
    main()