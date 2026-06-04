"""
arrival_rate.py — Arrival Rate Estimator
==========================================
Measures how many NEW vehicles enter each approach per unit time.

Design:
  - Rolling deque (in memory) for real-time λ_k(t)
  - Optional CSV log (on disk) for offline/historical analysis
  - O(1) amortized per frame

Works alongside QueueManager — both consume the same approach assignments.
"""

import csv
import os
from collections import defaultdict, deque


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_SECONDS = 15.0   # rolling window duration for λ computation


# ═══════════════════════════════════════════════════════════════════════════
# ARRIVAL RATE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════

class ArrivalRateEstimator:
    """
    Tracks arrival events per approach.

    - Real-time: rolling window deque → λ_k(t) in vehicles/second
    - Historical: optional CSV log for offline analysis
    """

    def __init__(self, fps=30.0, window_sec=WINDOW_SECONDS, log_enabled=False, log_path=None):
        """
        Args:
            fps:         Video frame rate
            window_sec:  Rolling window duration in seconds
            log_enabled: Whether to write arrival events to CSV
            log_path:    Path for CSV log file (default: arrivals_log.csv)
        """
        self.fps = fps
        self.window_frames = int(window_sec * fps)

        # Rolling window: {approach_label: deque of frame_numbers}
        self.arrivals = defaultdict(deque)

        # Track which vehicles we've already counted
        self.seen_vehicles = set()

        # CSV logging
        self.log_enabled = log_enabled
        self._log_file = None
        self._csv_writer = None

        if self.log_enabled:
            path = log_path or "arrivals_log.csv"
            self._log_file = open(path, "w", newline="")
            self._csv_writer = csv.writer(self._log_file)
            self._csv_writer.writerow(["frame", "timestamp_sec", "approach", "vehicle_id"])
            print(f"  [arrival_rate] CSV logging ON → {os.path.abspath(path)}")

    # ─────────────────────────────────────────────────────────────────────
    # CORE: record an arrival
    # ─────────────────────────────────────────────────────────────────────

    def record_arrival(self, tid, approach_label, frame_number):
        """
        Call when a vehicle is first assigned to an approach.
        Each vehicle is counted exactly once.
        """
        if tid in self.seen_vehicles:
            return

        self.seen_vehicles.add(tid)
        self.arrivals[approach_label].append(frame_number)

        # CSV log
        if self.log_enabled and self._csv_writer:
            timestamp = frame_number / self.fps
            self._csv_writer.writerow([frame_number, f"{timestamp:.2f}", approach_label, tid])

    # ─────────────────────────────────────────────────────────────────────
    # COMPUTE: λ_k(t) for a given approach
    # ─────────────────────────────────────────────────────────────────────

    def compute_rate(self, approach_label, current_frame):
        """
        Returns λ_k(t) in vehicles/second for the given approach.
        Trims expired entries from the deque (O(k) where k = expired, usually 0-1).
        """
        dq = self.arrivals[approach_label]
        cutoff = current_frame - self.window_frames

        # Trim old entries from left
        while dq and dq[0] < cutoff:
            dq.popleft()

        window_sec = self.window_frames / self.fps
        return len(dq) / window_sec

    def compute_all_rates(self, current_frame):
        """
        Returns {approach_label: λ_k(t)} for all approaches.
        """
        rates = {}
        for approach_label in list(self.arrivals.keys()):
            rates[approach_label] = self.compute_rate(approach_label, current_frame)
        return rates

    # ─────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────────

    def vehicle_exited(self, tid):
        """Clean up seen_vehicles when a vehicle disappears."""
        self.seen_vehicles.discard(tid)

    def close(self):
        """Flush and close the CSV log file."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._csv_writer = None
