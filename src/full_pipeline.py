"""
Main Tracking Pipeline with P(m|c) Learning
=============================================
Usage:
    python main_tracking.py --video control.mp4 --config control_config.json

What this does:
    1. Runs YOLO tracking on video
    2. Maps each vehicle position to a grid cell
    3. Detects when a vehicle crosses an exit line → labels its movement
    4. Updates P(m|c) for every cell that vehicle visited
    5. Visualizes heatmaps of learned intentions
"""

import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG & SETUP
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path):
    """Load exit line config from JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config: {list(config['exit_lines'].keys())} exit lines")
    return config


# ═══════════════════════════════════════════════════════════════════════════
# GRID CELL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_cell(x, y, cell_size):
    """Convert pixel coordinates to grid cell (cell_x, cell_y)."""
    return (int(x // cell_size), int(y // cell_size))


def draw_grid(frame, cell_size, color=(50, 50, 50), thickness=1):
    """Draw grid overlay on frame."""
    h, w = frame.shape[:2]
    for x in range(0, w, cell_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    for y in range(0, h, cell_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# LINE CROSSING DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def ccw(A, B, C):
    """Check if three points are counter-clockwise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, p3, p4):
    """
    Check if line segment p1-p2 intersects with line segment p3-p4.
    p1-p2: vehicle movement (previous position → current position)
    p3-p4: exit line endpoints
    """
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if d1 != d2 and d3 != d4:
        return True
    return False


def check_line_crossing(prev_pos, curr_pos, exit_lines):
    """
    Check if vehicle movement from prev_pos to curr_pos crosses any exit line.
    
    Args:
        prev_pos: (x, y) previous frame center point
        curr_pos: (x, y) current frame center point
        exit_lines: dict {"label": [[x1,y1], [x2,y2]], ...}
    
    Returns:
        label string if crossed, None otherwise
    """
    for label, line_pts in exit_lines.items():
        p3, p4 = line_pts[0], line_pts[1]
        if segments_intersect(prev_pos, curr_pos, p3, p4):
            return label
    return None


# ═══════════════════════════════════════════════════════════════════════════
# P(m|c) PROBABILISTIC MODEL
# ═══════════════════════════════════════════════════════════════════════════

class IntentionModel:
    """
    Learns P(m|c) — probability of movement m given cell c.
    
    Uses count-based updates with Laplace smoothing:
        P(m|c) = (N_c_m + alpha) / (N_c + K * alpha)
    
    where K = number of movement classes.
    """

    def __init__(self, movement_labels, alpha=1.0):
        """
        Args:
            movement_labels: list of movement names, e.g. ["left", "through", "right"]
            alpha: Laplace smoothing parameter
        """
        self.labels = movement_labels
        self.K = len(movement_labels)
        self.alpha = alpha

        # Per-cell counts: {(cell_x, cell_y): {"left": count, "through": count, ...}}
        self.cell_counts = defaultdict(lambda: defaultdict(float))
        # Per-cell total count
        self.cell_total = defaultdict(float)

        # Track how many vehicles have been labeled
        self.total_labeled = 0

    def update(self, trajectory_cells, movement_label):
        """
        Update model after a vehicle completes its trajectory.
        
        Args:
            trajectory_cells: list of (cell_x, cell_y) the vehicle visited
            movement_label: the exit label (e.g., "left")
        """
        # Deduplicate cells — only count each cell once per vehicle
        unique_cells = list(set(trajectory_cells))

        for cell in unique_cells:
            self.cell_counts[cell][movement_label] += 1
            self.cell_total[cell] += 1

        self.total_labeled += 1

    def get_prob(self, cell):
        """
        Get P(m|c) for a given cell.
        
        Returns:
            dict: {"left": prob, "through": prob, "right": prob}
        """
        total = self.cell_total[cell]
        probs = {}
        for m in self.labels:
            count = self.cell_counts[cell].get(m, 0)
            probs[m] = (count + self.alpha) / (total + self.K * self.alpha)
        return probs

    def get_dominant_movement(self, cell):
        """Get the most likely movement for a cell."""
        probs = self.get_prob(cell)
        if not probs:
            return None, 0
        best = max(probs, key=probs.get)
        return best, probs[best]

    def get_entropy(self, cell):
        """Compute entropy H(c) for a cell — lower means more confident."""
        probs = self.get_prob(cell)
        h = 0
        for p in probs.values():
            if p > 0:
                h -= p * np.log(p + 1e-10)
        return h

    def get_cell_support(self, cell):
        """Get total observation count for a cell."""
        return self.cell_total[cell]

    def summary(self):
        """Print model summary."""
        print(f"\n{'='*60}")
        print(f"  P(m|c) Model Summary")
        print(f"  Total labeled vehicles: {self.total_labeled}")
        print(f"  Cells with data: {len(self.cell_total)}")
        print(f"{'='*60}")

        # Show top cells by support
        sorted_cells = sorted(self.cell_total.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 15 cells by observation count:")
        for cell, count in sorted_cells[:15]:
            probs = self.get_prob(cell)
            entropy = self.get_entropy(cell)
            prob_str = " | ".join([f"{m}: {p:.2f}" for m, p in probs.items()])
            print(f"    Cell {cell}: N={count:.0f}  H={entropy:.3f}  [{prob_str}]")


# ═══════════════════════════════════════════════════════════════════════════
# HEATMAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Color map for each movement
MOVEMENT_COLORS = {
    # BGR format
    "zon1":    (0, 0, 255),     # Red
    "zone2":  (0, 255, 0),     # Green
    "zone3":   (255, 0, 0),     # Blue
}

# Fallback colors for custom labels
FALLBACK_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
]


def get_movement_color(label, idx=0):
    """Get color for a movement label."""
    if label in MOVEMENT_COLORS:
        return MOVEMENT_COLORS[label]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def draw_heatmap(frame, model, cell_size, min_support=2):
    """
    Draw P(m|c) heatmap overlay — each cell is colored by its dominant movement.
    Opacity reflects confidence (1 - entropy/max_entropy).
    """
    overlay = frame.copy()
    max_entropy = np.log(model.K)  # Maximum possible entropy

    for cell, total in model.cell_total.items():
        if total < min_support:
            continue

        cx, cy = cell
        x1, y1 = cx * cell_size, cy * cell_size
        x2, y2 = x1 + cell_size, y1 + cell_size

        best_movement, best_prob = model.get_dominant_movement(cell)
        entropy = model.get_entropy(cell)

        # Color by dominant movement
        color = get_movement_color(best_movement, model.labels.index(best_movement))

        # Opacity: higher confidence = more opaque
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0
        confidence = max(0.1, min(confidence, 0.7))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    return frame


def draw_exit_lines(frame, exit_lines):
    """Draw exit lines on frame."""
    for i, (label, pts) in enumerate(exit_lines.items()):
        p1, p2 = tuple(pts[0]), tuple(pts[1])
        color = get_movement_color(label, i)
        cv2.line(frame, p1, p2, color, 3)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(frame, label, (mx - 20, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Main Tracking Pipeline with P(m|c)")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--config", required=True, help="Path to zone config JSON")
    parser.add_argument("--model", default="../models/10_epoch.pt", help="YOLO model path")
    parser.add_argument("--cell_size", type=int, default=30, help="Grid cell size in pixels")
    parser.add_argument("--show_grid", action="store_true", help="Show grid overlay")
    parser.add_argument("--show_heatmap", action="store_true", default=True, help="Show P(m|c) heatmap")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    exit_lines = config["exit_lines"]
    cell_size = args.cell_size

    # Load YOLO model
    model = YOLO(args.model)
    class_list = model.names

    # Initialize intention model
    movement_labels = list(exit_lines.keys())
    intention = IntentionModel(movement_labels, alpha=1.0)
    print(f"Movement labels: {movement_labels}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.video}'")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} FPS, {total_frames} frames")

    # ── Tracking State ────────────────────────────────────────────────────
    # {track_id: [(cx, cy, frame_num, cell_x, cell_y), ...]}
    trajectory_history = defaultdict(list)

    # {track_id: (cx, cy)} — previous frame position for crossing detection
    prev_positions = {}

    # Set of track_ids that have already been labeled (crossed an exit line)
    labeled_ids = set()

    frame_number = 0

    print(f"\nRunning pipeline... Press 'q' to quit, 'h' to toggle heatmap\n")
    show_heatmap = args.show_heatmap

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Optional grid overlay
        if args.show_grid:
            frame = draw_grid(frame, cell_size)

        # Draw exit lines
        draw_exit_lines(frame, exit_lines)

        # Run YOLO tracking
        results = model.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu()

            for box, tid, cls_idx, conf in zip(boxes, track_ids, class_indices, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Grid cell
                cell = pixel_to_cell(cx, cy, cell_size)

                # Store trajectory
                trajectory_history[tid].append((cx, cy, frame_number, cell[0], cell[1]))

                # ── Check exit line crossing ──────────────────────────────
                if tid not in labeled_ids and tid in prev_positions:
                    prev = prev_positions[tid]
                    curr = (cx, cy)

                    crossed_label = check_line_crossing(prev, curr, exit_lines)

                    if crossed_label is not None:
                        # This vehicle just crossed an exit line!
                        labeled_ids.add(tid)

                        # Get all cells this vehicle visited
                        traj_cells = [(p[3], p[4]) for p in trajectory_history[tid]]

                        # Update P(m|c)
                        intention.update(traj_cells, crossed_label)

                        print(f"  [Frame {frame_number}] Track {tid} → {crossed_label} "
                              f"(visited {len(set(traj_cells))} cells) "
                              f"| Total labeled: {intention.total_labeled}")

                        # Draw crossing event
                        cv2.putText(frame, f"CROSSED: {crossed_label}",
                                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)

                # Update previous position
                prev_positions[tid] = (cx, cy)

                # ── Draw tracking info ────────────────────────────────────
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                label_text = f"ID:{tid}"
                if tid in labeled_ids:
                    # Show which exit it was labeled as
                    label_text += " ✓"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Show cell + P(m|c) for this cell
                probs = intention.get_prob(cell)
                support = intention.get_cell_support(cell)
                if support > 0:
                    prob_str = " ".join([f"{m[0].upper()}:{p:.1f}" for m, p in probs.items()])
                    cv2.putText(frame, prob_str, (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # ── Draw heatmap overlay ──────────────────────────────────────────
        if show_heatmap and intention.total_labeled > 0:
            frame = draw_heatmap(frame, intention, cell_size, min_support=2)

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames} | "
                    f"Cell: {cell_size}px | "
                    f"Labeled: {intention.total_labeled} | "
                    f"Tracking: {len(trajectory_history)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Cells with data: {len(intention.cell_total)} | "
                    f"Heatmap: {'ON' if show_heatmap else 'OFF'} (press 'h')",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("P(m|c) Learning Pipeline", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"  Heatmap: {'ON' if show_heatmap else 'OFF'}")

    # ═══════════════════════════════════════════════════════════════════════
    # CLEANUP & REPORT
    # ═══════════════════════════════════════════════════════════════════════
    cap.release()
    cv2.destroyAllWindows()

    # Print model summary
    intention.summary()

    # Save the learned model
    output_data = {
        "cell_size": cell_size,
        "total_labeled": intention.total_labeled,
        "movement_labels": movement_labels,
        "cells": {}
    }
    for cell, total in intention.cell_total.items():
        probs = intention.get_prob(cell)
        output_data["cells"][f"{cell[0]},{cell[1]}"] = {
            "support": total,
            "entropy": round(intention.get_entropy(cell), 4),
            "probabilities": {m: round(p, 4) for m, p in probs.items()}
        }

    model_path = Path(args.video).stem + "_pmc_model.json"
    with open(model_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Model saved to: {model_path}")


if __name__ == "__main__":
    main()