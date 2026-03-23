"""
queue_manager.py
=================
Handles:
  - Live queue counts per approach side (NB/SB/EB/WB)
  - Stop detection (vehicle waiting at red)
  - Backward credit assignment from partial trajectory
  - Soft intention distribution per waiting vehicle
  - Hard confirmation when vehicle exits
  - Queue state output for signal optimizer

Usage (in transition_tracker.py):
    from queue_manager import QueueManager, ExitZoneClassifier
"""

from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

STOP_FRAMES_THRESHOLD = 8    # frames in same cell → considered stopped
CREDIT_DECAY          = 0.85 # recent trajectory cells get more weight
MIN_TRAJ_FOR_CREDIT   = 3    # minimum cells needed before assigning credit
MODEL_WARMUP_FRAMES   = 50   # don't assign credit until model has this many transitions
APPROACH_MARGIN       = 2    # cells from edge counted as an approach entry


# ═══════════════════════════════════════════════════════════════════════════
# EXIT ZONE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class ExitZoneClassifier:
    """
    Maps a predicted path endpoint to a named exit label (Left/Through/Right).

    Setup example:
        classifier = ExitZoneClassifier()
        classifier.add_zone("Left",    [(0,5),(0,6),(0,7)])
        classifier.add_zone("Through", [(10,0),(11,0),(12,0)])
        classifier.add_zone("Right",   [(19,5),(19,6),(19,7)])
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


# ═══════════════════════════════════════════════════════════════════════════
# QUEUE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class QueueManager:
    """
    Full lifecycle manager for vehicles at an intersection.

    Lifecycle:
        1. Vehicle appears on approach edge  → register()
        2. Vehicle stops (same cell N frames) → assign_credit() called internally
        3. Vehicle exits frame               → confirm_exit() or remove()

    Queue state format (used by signal optimizer):
        {
            "NB": {"total": 5, "intentions": {"Left": 2.3, "Through": 1.8, "Right": 0.9}},
            "SB": {"total": 3, "intentions": {"Left": 0.5, "Through": 2.1, "Right": 0.4}},
            ...
        }
    """

    def __init__(self, grid_w, grid_h, exit_classifier=None):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.classifier = exit_classifier or ExitZoneClassifier()

        # Live counts
        self.queue            = defaultdict(int)           # {approach: total count}
        self.intention_queue  = defaultdict(                # {approach: {label: soft_count}}
                                    lambda: defaultdict(float))

        # Per-vehicle state
        self.tid_to_approach  = {}    # {tid: "NB"/"SB"/"EB"/"WB"}
        self.tid_to_credit    = {}    # {tid: {"Left": 0.6, "Through": 0.3, ...}}
        self.tid_to_status    = {}    # {tid: "moving" | "stopped" | "confirmed"}

        # Stop detection
        self.consecutive_same = defaultdict(int)   # {tid: frames in same cell}
        self.credit_assigned  = set()              # tids that already got credit

        # Stats
        self.total_registered  = 0
        self.total_confirmed   = 0
        self.total_removed     = 0

    # ───────────────────────────────────────────────────────────────────────
    # APPROACH DETECTION
    # ───────────────────────────────────────────────────────────────────────

    def get_approach(self, cell):
        """Determine approach side from entry cell. Returns 'NB/SB/EB/WB' or 'UNKNOWN'."""
        cx, cy = cell
        m = APPROACH_MARGIN
        if cy <= m:                    return "SB"
        if cy >= self.grid_h - m:      return "NB"
        if cx <= m:                    return "EB"
        if cx >= self.grid_w - m:      return "WB"
        return "UNKNOWN"

    # ───────────────────────────────────────────────────────────────────────
    # REGISTRATION
    # ───────────────────────────────────────────────────────────────────────

    def register(self, tid, first_cell):
        """
        Call once on first detection.
        Only registers vehicles entering from a real approach edge.
        Returns approach label or None if mid-intersection appearance.
        """
        approach = self.get_approach(first_cell)
        if approach == "UNKNOWN":
            return None

        self.tid_to_approach[tid] = approach
        self.tid_to_credit[tid]   = {}
        self.tid_to_status[tid]   = "moving"
        self.queue[approach]     += 1
        self.total_registered    += 1
        return approach

    # ───────────────────────────────────────────────────────────────────────
    # STOP DETECTION + CREDIT ASSIGNMENT
    # ───────────────────────────────────────────────────────────────────────

    def update(self, tid, current_cell, prev_cell, trajectory_history, trans):
        """
        Call every frame for every tracked vehicle.
        Internally handles stop detection and triggers credit assignment.

        Args:
            tid               : track id
            current_cell      : (cx, cy) this frame
            prev_cell         : (cx, cy) last frame (or None)
            trajectory_history: the full defaultdict(list) from main
            trans             : TransitionModel instance
        """
        if tid not in self.tid_to_approach:
            return   # not registered (mid-intersection appearance)

        # ── Stop detection ────────────────────────────────────────────────
        if prev_cell is not None and current_cell == prev_cell:
            self.consecutive_same[tid] += 1
        else:
            self.consecutive_same[tid] = 0
            # Vehicle moved again after stopping → re-evaluate credit
            if tid in self.credit_assigned:
                self.credit_assigned.discard(tid)
                self._remove_credit(tid)
                self.tid_to_status[tid] = "moving"

        just_stopped = (
            self.consecutive_same[tid] == STOP_FRAMES_THRESHOLD
            and tid not in self.credit_assigned
            and trans.total_transitions >= MODEL_WARMUP_FRAMES
        )

        if just_stopped:
            credit = self._compute_credit(tid, trajectory_history, trans)
            if credit:
                self._apply_credit(tid, credit)
                self.credit_assigned.add(tid)
                self.tid_to_status[tid] = "stopped"

    def _compute_credit(self, tid, trajectory_history, trans):
        """
        Look back at partial trajectory and compute soft intention distribution.

        Logic:
          - Take the last N unique cells from trajectory
          - For each cell, run predict_path() and classify endpoint
          - Weight recent cells more heavily (exponential decay)
          - Normalize to sum to 1.0

        Returns:
            {"Left": 0.6, "Through": 0.3, "Right": 0.1} or {}
        """
        traj = trajectory_history.get(tid, [])
        if len(traj) < MIN_TRAJ_FOR_CREDIT:
            return {}

        if not self.classifier.is_configured():
            return {}

        # Extract unique cells in order (remove consecutive duplicates)
        unique_cells = []
        for entry in traj:
            cell = (entry[3], entry[4])
            if not unique_cells or unique_cells[-1] != cell:
                unique_cells.append(cell)

        # Use last 10 unique cells, most recent = highest weight
        recent = unique_cells[-10:]

        credit = defaultdict(float)
        weight = 1.0
        # Iterate from most recent → oldest
        for cell in reversed(recent):
            path  = trans.predict_path(cell, steps=20)
            label = self.classifier.classify(path)
            if label != "UNKNOWN":
                credit[label] += weight
            weight *= CREDIT_DECAY

        if not credit:
            return {}

        # Normalize
        total = sum(credit.values())
        return {k: round(v / total, 4) for k, v in credit.items()}

    def _apply_credit(self, tid, credit):
        """Add soft intention counts to queue."""
        approach = self.tid_to_approach[tid]
        self.tid_to_credit[tid] = credit
        for label, prob in credit.items():
            self.intention_queue[approach][label] += prob

    def _remove_credit(self, tid):
        """Remove previously applied soft credit (vehicle moved again)."""
        approach = self.tid_to_approach.get(tid)
        old_credit = self.tid_to_credit.get(tid, {})
        if approach:
            for label, prob in old_credit.items():
                self.intention_queue[approach][label] = max(
                    0.0, self.intention_queue[approach][label] - prob
                )
        self.tid_to_credit[tid] = {}

    # ───────────────────────────────────────────────────────────────────────
    # EXIT CONFIRMATION
    # ───────────────────────────────────────────────────────────────────────

    def confirm_exit(self, tid, actual_label):
        """
        Call when vehicle reaches a known exit zone.
        Replaces soft credit with hard +1 confirmed count.
        Updates model accuracy stats.

        Args:
            tid          : track id
            actual_label : "Left" / "Through" / "Right" (ground truth)
        """
        if tid not in self.tid_to_approach:
            return

        approach = self.tid_to_approach[tid]

        # Remove soft credit
        self._remove_credit(tid)

        # Add confirmed hard count
        if actual_label != "UNKNOWN":
            self.intention_queue[approach][actual_label] += 1.0

        self.tid_to_status[tid] = "confirmed"
        self.total_confirmed += 1

        # Clean up
        self._cleanup(tid)

    def remove(self, tid):
        """
        Call when vehicle disappears without confirmed exit
        (left frame edge, occlusion, tracker loss).
        Removes from queue and cleans up.
        """
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

    # ───────────────────────────────────────────────────────────────────────
    # SIGNAL OPTIMIZER INTERFACE
    # ───────────────────────────────────────────────────────────────────────

    def get_queue_state(self):
        """
        Returns full queue state for signal optimizer.

        Format:
            {
                "NB": {
                    "total": 5,
                    "intentions": {"Left": 2.3, "Through": 1.8, "Right": 0.9}
                },
                ...
            }
        """
        state = {}
        for approach in ["NB", "SB", "EB", "WB"]:
            total      = self.queue.get(approach, 0)
            intentions = dict(self.intention_queue.get(approach, {}))
            state[approach] = {
                "total":      total,
                "intentions": intentions
            }
        return state

    def get_pressure(self):
        """
        Simple pressure score per approach.
        Returns {approach: score} — higher = needs green more urgently.
        Can be extended to weight certain intentions more (e.g. left turns).
        """
        pressure = {}
        for approach, data in self.get_queue_state().items():
            pressure[approach] = data["total"]
        return pressure

    # ───────────────────────────────────────────────────────────────────────
    # VISUALIZATION HELPERS
    # ───────────────────────────────────────────────────────────────────────

    def draw_hud(self, frame):
        """Draw live queue panel on frame (top-right corner)."""
        import cv2
        h, w = frame.shape[:2]
        panel_x = w - 300
        panel_y = 10
        panel_h = 130

        # Background
        cv2.rectangle(frame, (panel_x - 5, panel_y),
                      (w - 5, panel_y + panel_h), (0, 0, 0), -1)
        cv2.putText(frame, "QUEUE STATE", (panel_x, panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        y = panel_y + 38
        for approach in ["NB", "SB", "EB", "WB"]:
            total      = self.queue.get(approach, 0)
            intentions = self.intention_queue.get(approach, {})
            color      = (0, 255, 0) if total > 0 else (80, 80, 80)

            # e.g.  NB:  4  L:2.1 T:1.5 R:0.4
            intent_str = "  ".join(
                f"{k[0]}:{v:.1f}" for k, v in sorted(intentions.items())
                if v > 0.05
            ) or "waiting for data..."

            cv2.putText(frame, f"{approach}: {total:>2}  {intent_str}",
                        (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 22

        return frame

    def draw_vehicle_label(self, frame, tid, x1, y1, y2, current_cell):
        """Draw per-vehicle label showing approach + credit distribution."""
        import cv2
        approach = self.tid_to_approach.get(tid, "?")
        status   = self.tid_to_status.get(tid, "?")
        credit   = self.tid_to_credit.get(tid, {})

        status_color = {
            "moving":    (0, 255, 255),
            "stopped":   (0, 165, 255),
            "confirmed": (0, 255, 0),
        }.get(status, (200, 200, 200))

        label = f"ID:{tid} [{approach}] {status}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        if credit:
            credit_str = " ".join(f"{k[0]}:{v:.0%}" for k, v in credit.items())
            cv2.putText(frame, credit_str, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        return frame

    # ───────────────────────────────────────────────────────────────────────
    # DEBUG / STATS
    # ───────────────────────────────────────────────────────────────────────

    def print_status(self):
        print(f"\n  ── Queue Status ──────────────────────────────")
        state = self.get_queue_state()
        for approach, data in state.items():
            total = data["total"]
            intents = data["intentions"]
            intent_str = "  ".join(
                f"{k}:{v:.1f}" for k, v in sorted(intents.items()) if v > 0.05
            ) or "no data"
            print(f"    {approach}: {total:>3} cars  │  {intent_str}")
        print(f"  Registered: {self.total_registered} | "
              f"Confirmed: {self.total_confirmed} | "
              f"Removed: {self.total_removed}")
        print(f"  ─────────────────────────────────────────────")
