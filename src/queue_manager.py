"""
queue_manager.py — Lane-Free Queue Management
================================================
Decoupled design:
  1. Approaches are discovered by clustering trajectory start points
     — no hardcoded NB/SB/EB/WB, any number of approaches possible
  2. Queue detection is per approach (speed-based stop detection)
  3. Exit prediction is handled independently by the LSTM

No lane detection, no manual polygons — everything learned from data.
"""

import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

VELOCITY_THRESHOLD  = 4.0    # m/s — below this, vehicle is considered queued
SMOOTHING_FRAMES    = 5      # number of frames to average velocity over
ASSUMED_INTERSECTION_DIST = 30.0  # meters — assumed real-world distance across intersection
INTERSECTION_MARGIN = 0.35   # center 35% of grid = intersection (excluded)
APPROACH_RADIUS     = 8    # clustering radius for start-point grouping
MIN_APPROACH_COUNT  = 21     # minimum trajectories to form an approach
MIN_TRAJ_LENGTH     = 5      # minimum trajectory cells to count for approach


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-GENERATED COLORS (for any number of approaches)
# ═══════════════════════════════════════════════════════════════════════════

APPROACH_COLORS = [
    (0, 255, 128),   # Green
    (0, 128, 255),   # Orange-blue
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (128, 255, 0),   # Lime
    (0, 255, 255),   # Yellow
    (255, 128, 0),   # Blue-ish
    (128, 0, 255),   # Purple
    (255, 0, 128),   # Pink
    (0, 200, 200),   # Teal
]


# ═══════════════════════════════════════════════════════════════════════════
# QUEUE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class QueueManager:
    """
    Lane-free queue management — decoupled from exit prediction.

    Responsibilities:
        - Discover approaches by clustering trajectory start points
        - Assign vehicles to nearest approach on entry
        - Detect queued (stopped) vehicles per approach
        - Report queue counts per approach

    Does NOT handle exit prediction — that's the LSTM's job.
    """

    def __init__(self, grid_w, grid_h, fps=30.0, intersection_dist=None, intersection_margin=None):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.fps = fps
        self.intersection_dist = intersection_dist or ASSUMED_INTERSECTION_DIST
        self.intersection_margin = intersection_margin if intersection_margin is not None else INTERSECTION_MARGIN

        # Approach data: [{"label": "A0", "center": (cx,cy), "cells": set(), "count": N}, ...]
        self.approaches = []
        # Quick lookup: {"A0": set of (cx,cy), ...}
        self.approach_rois = {}

        # Per-vehicle state — velocity-based stop detection
        self.vehicle_pixels    = {}    # {tid: [(px, py), ...]} recent pixel positions
        self.vehicle_velocity  = {}    # {tid: float} smoothed velocity in m/s
        self.vehicle_approach  = {}    # {tid: "A0"/"A1"/...}
        self.vehicle_cells     = {}    # {tid: [(cx,cy), ...]}

        # Calibration: pixels → meters (set after approaches are discovered)
        self.pixels_to_meters = None

        # Stats
        self.total_completed = 0

    # ────────────────────────
    # ─────────────────────────────────────────────
    # SETUP: auto-discover approaches by clustering start points
    # ─────────────────────────────────────────────────────────────────────

    def auto_discover_approaches(self, trajectories, grid_cols,
                                  radius=APPROACH_RADIUS,
                                  min_count=MIN_APPROACH_COUNT):
        """
        Discover approach regions by clustering trajectory start points.

        1. Collect start cells from all trajectories
        2. Greedy cluster:
        as one approach
        3. For each approach, collect the first 1/3 of member trajectories'
           cells (outside intersection) as the ROI
        """
        # Collect start cells → trajectory IDs (filter short trajectories)
        start_counts = defaultdict(list)  # {(cx,cy): [tid1, tid2, ...]}
        for tid, traj in trajectories.items():
            cells_list = traj["cells"]
            if len(cells_list) < MIN_TRAJ_LENGTH:
                continue
            start = tuple(cells_list[0])
            start_counts[start].append(tid)

        # Greedy clustering (same approach as endpoint clustering)
        sorted_starts = sorted(start_counts.items(),
                               key=lambda x: len(x[1]), reverse=True)
        used_cells = set()
        clusters = []

        for cell, tids in sorted_starts:
            if cell in used_cells:
                continue

            cluster_cells = set()
            cluster_tids = []

            for other_cell, other_tids in sorted_starts:
                if other_cell in used_cells:
                    continue
                dist = abs(cell[0] - other_cell[0]) + abs(cell[1] - other_cell[1])
                if dist <= radius:
                    cluster_cells.add(other_cell)
                    cluster_tids.extend(other_tids)

            if len(cluster_tids) >= min_count:
                used_cells.update(cluster_cells)
                cx = np.mean([c[0] for c in cluster_cells])
                cy = np.mean([c[1] for c in cluster_cells])
                clusters.append({
                    "center": (cx, cy),
                    "start_cells": cluster_cells,
                    "tids": cluster_tids,
                    "count": len(cluster_tids),
                })

        # Filter out clusters entirely inside the intersection
        # A cluster is valid if ANY of its start cells are outside the intersection
        clusters = [cl for cl in clusters
                    if any(not self._in_intersection(c[0], c[1])
                           for c in cl["start_cells"])]

        # Keep only clusters that have cells near frame edges
        # Real approaches always have vehicles entering from the edge,
        # so at least one start cell must be within edge_margin of a border.
        edge_margin = 0.15
        def has_edge_cell(cl):
            for c in cl["start_cells"]:
                if (c[0] < self.grid_w * edge_margin or
                    c[0] > self.grid_w * (1 - edge_margin) or
                    c[1] < self.grid_h * edge_margin or
                    c[1] >= self.grid_h * (1 - edge_margin)):
                    return True
            return False
        clusters = [cl for cl in clusters if has_edge_cell(cl)]

        # Merge clusters on the same side of the intersection.
        # With longer videos, a single approach can split into multiple
        # clusters. Merge any two whose centers are on the same side.
        cx_mid = self.grid_w / 2
        cy_mid = self.grid_h / 2

        def _side(cl):
            cx, cy = cl["center"]
            dx, dy = abs(cx - cx_mid), abs(cy - cy_mid)
            if dy > dx:
                return "top" if cy < cy_mid else "bottom"
            else:
                return "left" if cx < cx_mid else "right"

        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if _side(clusters[i]) == _side(clusters[j]):
                        ci, cj = clusters[i], clusters[j]
                        total = ci["count"] + cj["count"]
                        ci["center"] = (
                            (ci["center"][0] * ci["count"] + cj["center"][0] * cj["count"]) / total,
                            (ci["center"][1] * ci["count"] + cj["center"][1] * cj["count"]) / total,
                        )
                        ci["start_cells"] = ci["start_cells"] | cj["start_cells"]
                        ci["tids"] = ci["tids"] + cj["tids"]
                        ci["count"] = total
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break

        # Build ROIs from trajectory paths
        self.approaches = []
        self.approach_rois = {}

        for i, cl in enumerate(clusters):
            label = f"A{i}"
            roi_cells = set()

            for tid_str in cl["tids"]:
                traj = trajectories.get(str(tid_str), trajectories.get(tid_str))
                if traj is None:
                    continue
                cells_list = traj["cells"]
                n = max(len(cells_list) // 4, 3)
                for cell_raw in cells_list[:n]:
                    cell = tuple(cell_raw)
                    if not self._in_intersection(cell[0], cell[1]):
                        roi_cells.add(cell)

            approach = {
                "label": label,
                "center": cl["center"],
                "cells": roi_cells,
                "count": cl["count"],
            }
            self.approaches.append(approach)
            self.approach_rois[label] = roi_cells

            print(f"  {label}: center=({cl['center'][0]:.1f}, {cl['center'][1]:.1f}), "
                  f"{cl['count']} trajectories, {len(roi_cells)} ROI cells")

        print(f"  Total: {len(self.approaches)} approaches discovered")

        # Auto-calibrate pixels→meters from approach spacing
        self._calibrate_scale(cell_size=None)

    def _calibrate_scale(self, cell_size=None):
        """
        Compute pixels_to_meters by finding the two farthest approach centers
        and dividing the assumed real-world distance by their pixel distance.
        """
        if len(self.approaches) < 2:
            print("  [calibration] <2 approaches, using fallback scale")
            self.pixels_to_meters = 0.05  # rough fallback
            return

        # Find the two farthest approach centers (in grid coords)
        max_dist = 0
        pair = (0, 1)
        for i in range(len(self.approaches)):
            for j in range(i + 1, len(self.approaches)):
                ci = self.approaches[i]["center"]
                cj = self.approaches[j]["center"]
                d = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                if d > max_dist:
                    max_dist = d
                    pair = (i, j)

        # max_dist is in grid-cell units; we need it in pixels
        # Grid center coords are already in cell units, so pixel dist = max_dist * cell_size
        # But we don't know cell_size here — we store grid coords, so use grid distance directly
        # pixels_to_meters = real_world_dist / pixel_dist
        # Since we track pixel positions directly, we need pixel distance between approach centers
        # Approach centers are in grid coords → we'll convert when we have cell_size
        self._approach_pixel_dist = max_dist  # in grid-cell units for now
        self.pixels_to_meters = self.intersection_dist / max_dist if max_dist > 0 else 0.05

        a_i = self.approaches[pair[0]]["label"]
        a_j = self.approaches[pair[1]]["label"]
        print(f"  [calibration] {a_i} ↔ {a_j}: {max_dist:.1f} grid-cells = "
              f"{self.intersection_dist}m assumed → "
              f"{self.pixels_to_meters:.4f} m/grid-cell")

    def calibrate_with_cell_size(self, cell_size):
        """
        Refine calibration once we know the cell_size in pixels.
        Call this from the main loop after setup.
        """
        if hasattr(self, '_approach_pixel_dist') and self._approach_pixel_dist > 0:
            pixel_dist = self._approach_pixel_dist * cell_size
            self.pixels_to_meters = self.intersection_dist / pixel_dist
            print(f"  [calibration] refined: {pixel_dist:.0f}px = "
                  f"{self.intersection_dist}m → "
                  f"{self.pixels_to_meters:.5f} m/px")

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _in_intersection(self, cx, cy):
        m = self.intersection_margin
        return (self.grid_w * m < cx < self.grid_w * (1 - m) and
                self.grid_h * m < cy < self.grid_h * (1 - m))

    def _get_approach(self, cell):
        """
        Which approach does this cell belong to?
        1. Check if cell is in exactly one ROI → use that
        2. If in multiple ROIs → pick nearest approach center
        3. If in no ROI → pick nearest approach center (if outside intersection)
        """
        cell = tuple(cell)
        
        cx, cy = cell
        if self._in_intersection(cx, cy):
            return None

        if not self.approaches:
            return None

        # Check which ROIs contain this cell
        matching = [a["label"] for a in self.approaches if cell in a["cells"]]

        if len(matching) == 1:
            return matching[0]

        # Pick nearest approach center (among matching, or all if none match)
        candidates = matching if matching else [a["label"] for a in self.approaches]
        center_map = {a["label"]: a["center"] for a in self.approaches}

        def dist_to_center(label):
            ac = center_map[label]
            return (cx - ac[0])**2 + (cy - ac[1])**2

        best = min(candidates, key=dist_to_center)

        # If cell wasn't in any ROI, only assign if close enough
        # Otherwise vehicles on the opposite side get wrongly assigned
        if not matching:
            max_dist = APPROACH_RADIUS * 3
            if dist_to_center(best) > max_dist**2:
                return None

        return best

    # ─────────────────────────────────────────────────────────────────────
    # PER-FRAME TRACKING
    # ─────────────────────────────────────────────────────────────────────

    def track_vehicle(self, tid, current_cell, prev_cell, pixel_pos=None):
        """
        Call every frame for every visible vehicle.
        Handles: approach assignment, velocity-based stop detection, trajectory recording.

        Args:
            pixel_pos: (px, py) pixel coordinates of vehicle center.
                       Required for velocity calculation.
        """
        cell = tuple(current_cell)

        # Velocity-based stop detection
        if pixel_pos is not None:
            if tid not in self.vehicle_pixels:
                self.vehicle_pixels[tid] = []
            self.vehicle_pixels[tid].append(pixel_pos)
            # Keep only last SMOOTHING_FRAMES positions
            if len(self.vehicle_pixels[tid]) > SMOOTHING_FRAMES:
                self.vehicle_pixels[tid] = self.vehicle_pixels[tid][-SMOOTHING_FRAMES:]
            self._update_velocity(tid)

        # Record trajectory (unique consecutive cells only)
        if tid not in self.vehicle_cells:
            self.vehicle_cells[tid] = [cell]
            approach = self._get_approach(cell)
            if approach:
                self.vehicle_approach[tid] = approach
        else:
            if self.vehicle_cells[tid][-1] != cell:
                self.vehicle_cells[tid].append(cell)
            # Only assign approach if not already assigned
            if tid not in self.vehicle_approach:
                approach = self._get_approach(cell)
                if approach:
                    self.vehicle_approach[tid] = approach

    def _update_velocity(self, tid):
        """Compute smoothed velocity (m/s) from recent pixel positions."""
        positions = self.vehicle_pixels[tid]
        if len(positions) < 2:
            self.vehicle_velocity[tid] = 0.0
            return

        # Average displacement over the stored positions
        total_disp = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_disp += np.sqrt(dx**2 + dy**2)

        avg_disp_per_frame = total_disp / (len(positions) - 1)  # pixels/frame
        pixels_per_sec = avg_disp_per_frame * self.fps

        scale = self.pixels_to_meters if self.pixels_to_meters else 0.05
        self.vehicle_velocity[tid] = pixels_per_sec * scale

    def vehicle_exited(self, tid):
        """Call when a vehicle disappears. Cleans up state."""
        cells = self.vehicle_cells.get(tid)
        if cells and len(cells) >= 3:
            self.total_completed += 1
        self._cleanup(tid)

    def _cleanup(self, tid):
        self.vehicle_pixels.pop(tid, None)
        self.vehicle_velocity.pop(tid, None)
        self.vehicle_approach.pop(tid, None)
        self.vehicle_cells.pop(tid, None)

    def is_queued(self, tid):
        return self.vehicle_velocity.get(tid, float('inf')) < VELOCITY_THRESHOLD

    def get_velocity(self, tid):
        """Get current smoothed velocity in m/s for a vehicle."""
        return self.vehicle_velocity.get(tid, 0.0)

    # ─────────────────────────────────────────────────────────────────────
    # QUEUE STATE OUTPUT — approach queues only, no exit mixing
    # ─────────────────────────────────────────────────────────────────────

    def compute_state(self, active_vehicle_cells):
        """
        Compute queue counts per approach.

        Args:
            active_vehicle_cells: {tid: (cx, cy)} for all visible vehicles

        Returns:
            {"A0": {"queue": 3, "total": 5}, "A1": {"queue": 6, "total": 8}, ...}
        """
        state = {}

        for approach_label in self.approach_rois:
            approach_vehs = []
            queued_vehs   = []

            for tid, cell in active_vehicle_cells.items():
                if self.vehicle_approach.get(tid) != approach_label:
                    continue
                approach_vehs.append(tid)
                if self.is_queued(tid):
                    queued_vehs.append(tid)

            state[approach_label] = {
                "queue": len(queued_vehs),
                "total": len(approach_vehs),
            }

        return state

    # ─────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────

    def _get_color(self, label):
        """Get a consistent color for an approach label."""
        idx = int(label[1:]) if label.startswith("A") and label[1:].isdigit() else 0
        return APPROACH_COLORS[idx % len(APPROACH_COLORS)]

    def draw_state(self, frame, state, cell_size, exit_predictions=None):
        """Draw queue state HUD on frame."""
        import cv2
        h, w = frame.shape[:2]
        px = w - 360
        py = 10

        line_h = 22
        lines_per_approach = 1
        if exit_predictions:
            lines_per_approach += 1
        panel_h = 24 + lines_per_approach * line_h * len(state)
        cv2.rectangle(frame, (px - 5, py), (w - 5, py + panel_h), (0, 0, 0), -1)
        cv2.putText(frame, "QUEUE STATE", (px, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        y = py + 36

        for approach, data in sorted(state.items()):
            q = data["queue"]
            total = data["total"]
            color = self._get_color(approach)

            cv2.putText(frame, f"{approach}: {q} queued / {total} total",
                        (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += line_h

            if exit_predictions:
                exit_counts = defaultdict(int)
                for tid in self._get_approach_tids(approach):
                    if tid in exit_predictions:
                        exit_counts[exit_predictions[tid]["label"]] += 1
                if exit_counts:
                    parts = [f"{lbl}:{cnt}" for lbl, cnt in sorted(exit_counts.items())]
                    cv2.putText(frame, f"  exits: {' '.join(parts)}",
                                (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
                else:
                    cv2.putText(frame, f"  exits: predicting...",
                                (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
                y += line_h

        return frame

    def _get_approach_tids(self, approach_label):
        """Return list of tids assigned to this approach."""
        return [tid for tid, appr in self.vehicle_approach.items()
                if appr == approach_label]

    def draw_approach_rois(self, frame, cell_size, alpha=0.12):
        """Draw colored overlay for each approach ROI and intersection exclusion zone."""
        import cv2
        overlay = frame.copy()

        # Draw approach ROIs
        for a in self.approaches:
            color = self._get_color(a["label"])
            for (cx, cy) in a["cells"]:
                px1 = cx * cell_size
                py1 = cy * cell_size
                cv2.rectangle(overlay, (px1, py1),
                              (px1 + cell_size, py1 + cell_size), color, -1)

        # Draw intersection exclusion zone
        m = self.intersection_margin
        ix1 = int(self.grid_w * m * cell_size)
        iy1 = int(self.grid_h * m * cell_size)
        ix2 = int(self.grid_w * (1 - m) * cell_size)
        iy2 = int(self.grid_h * (1 - m) * cell_size)
        cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (0, 0, 200), -1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw border and label on top
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 0, 200), 2)
        cv2.putText(frame, 
        "EXCLUDED (overlap zone)", (ix1 + 4, iy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        # Draw approach labels at their centers
        for a in self.approaches:
            cx, cy = a["center"]
            tx = int(cx * cell_size)
            ty = int(cy * cell_size)
            color = self._get_color(a["label"])
            cv2.putText(frame, f"{a['label']} ({a['count']})",
                        (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame
