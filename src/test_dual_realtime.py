"""
Dual-Model Real-Time Prediction (LSTM + Markov Chain)
======================================================
Usage:
    python test_dual_realtime.py \
        --video data/videos/testing_part.mp4 \
        --lstm_model models/lstm/training_part_trajectories_lstm_model.pt \
        --markov_model models/training_part_transitions_markov_model.json \
        --yolo ../models/best.pt

Runs YOLO tracking and shows side-by-side predictions from:
    • LSTM  — learns from partial trajectory sequences
    • Markov — uses P(exit_zone | current_cell) from Monte Carlo simulation

Controls:
    SPACE : Pause / Resume
    'q'   : Quit
    'g'   : Toggle grid
    'm'   : Cycle display mode (Both → LSTM only → Markov only → Both)
"""

import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from ultralytics import YOLO

from grid_utils import cell_to_id, id_to_cell


# ═══════════════════════════════════════════════════════════════════════════
# LSTM Model (same architecture as training)
# ═══════════════════════════════════════════════════════════════════════════

class TurnPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed)
        last_hidden = hidden[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_cell(x, y, cell_size):
    return (int(x // cell_size), int(y // cell_size))


def trajectory_to_features(cells):
    """Convert cell sequence to (x, y, dx, dy) features — same as training."""
    features = []
    for i, (x, y) in enumerate(cells):
        if i == 0:
            dx, dy = 0.0, 0.0
        else:
            dx = x - cells[i-1][0]
            dy = y - cells[i-1][1]
        features.append([x / 40.0, y / 22.0, dx / 5.0, dy / 5.0])
    return np.array(features, dtype=np.float32)


def draw_grid(frame, cell_size, color=(100, 100, 100), thickness=1):
    h, w = frame.shape[:2]
    for x in range(0, w, cell_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    for y in range(0, h, cell_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cx, cy = x // cell_size, y // cell_size
            cv2.putText(frame, f"{cx},{cy}", (x+2, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    return frame


# Color per exit zone
EXIT_COLORS = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
]


# ═══════════════════════════════════════════════════════════════════════════
# LSTM PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class LSTMPredictor:
    def __init__(self, model_path, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.label_map = checkpoint["label_map"]
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.clusters = checkpoint["clusters"]
        self.cell_size = checkpoint["cell_size"]
        self.wait_priors = {}
        if "wait_priors" in checkpoint:
            for item in checkpoint["wait_priors"]:
                self.wait_priors[tuple(item["cell"])] = item["probs"]
        config = checkpoint["model_config"]

        self.model = TurnPredictor(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"]
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.min_steps = 3

        print(f"  LSTM loaded: {len(self.label_map)} exit classes, cell_size={self.cell_size}")

    def predict(self, cell_sequence):
        """Returns (label, confidence, all_probs) or (None, 0, {})."""
        if len(cell_sequence) < self.min_steps:
            if len(cell_sequence) > 0:
                start_cell = tuple(cell_sequence[0])
                if start_cell in self.wait_priors:
                    probs = self.wait_priors[start_cell]
                    pred_label = max(probs, key=probs.get)
                    return pred_label, probs[pred_label], probs
            return None, 0.0, {}

        features = trajectory_to_features(cell_sequence)
        seq = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        length = torch.tensor([len(cell_sequence)], dtype=torch.long)

        with torch.no_grad():
            output = self.model(seq, length)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        pred_idx = np.argmax(probs)
        pred_label = self.inv_label_map[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs = {self.inv_label_map[i]: float(p) for i, p in enumerate(probs)}
        return pred_label, confidence, all_probs


# ═══════════════════════════════════════════════════════════════════════════
# MARKOV PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class MarkovPredictor:
    def __init__(self, model_path):
        with open(model_path, 'r') as f:
            data = json.load(f)

        self.cell_size = data["cell_size"]
        self.grid_cols = data["grid_cols"]
        self.exit_zones = data["exit_zones"]

        # cell_predictions: {linear_cell_id_str: {exit_label: prob, ...}}
        self.cell_predictions = data["cell_predictions"]

        # flow_chains: {linear_cell_id_str: {"next": [col, row], "probability": float}}
        self.flow_chains = data.get("flow_chains", {})

        print(f"  Markov loaded: {len(self.cell_predictions)} cells with predictions, "
              f"{len(self.exit_zones)} exit zones, cell_size={self.cell_size}")

    def predict(self, cell_sequence):
        """
        Predict exit from current cell using pre-computed P(exit | cell).

        Returns (label, confidence, all_probs) or (None, 0, {}).
        """
        if not cell_sequence:
            return None, 0.0, {}

        # Use the LATEST cell the vehicle is in
        current_cell = cell_sequence[-1]
        cell_id = str(cell_to_id(current_cell[0], current_cell[1], self.grid_cols))

        if cell_id not in self.cell_predictions:
            return None, 0.0, {}

        probs = self.cell_predictions[cell_id]

        # Filter out "unknown" for prediction label but keep it in probs
        real_probs = {k: v for k, v in probs.items() if k != "unknown"}
        if not real_probs:
            return None, 0.0, probs

        pred_label = max(real_probs, key=real_probs.get)
        confidence = real_probs[pred_label]

        return pred_label, confidence, probs

    def get_predicted_path(self, start_cell, steps=8):
        """Follow the most-likely-next chain from the Markov model."""
        path = [start_cell]
        current = start_cell
        for _ in range(steps):
            cell_id = str(cell_to_id(current[0], current[1], self.grid_cols))
            if cell_id not in self.flow_chains:
                break
            next_info = self.flow_chains[cell_id]
            next_cell = tuple(next_info["next"])
            if next_cell == current:  # avoid loops
                break
            path.append(next_cell)
            current = next_cell
        return path


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_exit_zones(frame, exit_zones, cell_size, alpha=0.35):
    """Draw exit zones as colored overlays."""
    overlay = frame.copy()
    for i, zone in enumerate(exit_zones):
        color = EXIT_COLORS[i % len(EXIT_COLORS)]
        for c in zone["cells"]:
            px1 = int(c[0] * cell_size)
            py1 = int(c[1] * cell_size)
            px2 = px1 + cell_size
            py2 = py1 + cell_size
            cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        center_x = int(zone["center"][0] * cell_size + cell_size // 2)
        center_y = int(zone["center"][1] * cell_size + cell_size // 2)
        text = zone["label"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (center_x - 2, center_y - th - 2),
                      (center_x + tw + 2, center_y + 2), (0, 0, 0), -1)
        cv2.putText(overlay, text, (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_markov_path(frame, path, cell_size, color=(0, 200, 200)):
    """Draw predicted Markov path as a dashed line."""
    if len(path) < 2:
        return frame
    for i in range(len(path) - 1):
        p1 = (int(path[i][0] * cell_size + cell_size // 2),
              int(path[i][1] * cell_size + cell_size // 2))
        p2 = (int(path[i+1][0] * cell_size + cell_size // 2),
              int(path[i+1][1] * cell_size + cell_size // 2))
        alpha = 1.0 - (i / len(path))
        c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.line(frame, p1, p2, c, 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

DISPLAY_MODES = ["both", "lstm_only", "markov_only"]

def main():
    parser = argparse.ArgumentParser(description="Dual-Model Real-Time Prediction")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--lstm_model", required=True, help="Path to LSTM model (.pt)")
    parser.add_argument("--markov_model", required=True, help="Path to Markov model (.json)")
    parser.add_argument("--yolo", default="../models/best.pt", help="YOLO model path")
    parser.add_argument("--cell_size", type=int, default=50, help="Display grid cell size")
    parser.add_argument("--output", help="Path to save output video")
    args = parser.parse_args()

    # ── Load models ───────────────────────────────────────────────────────
    print("\n── Loading models ──")
    lstm = LSTMPredictor(args.lstm_model)
    markov = MarkovPredictor(args.markov_model)

    # Use the LSTM model's cell_size for cell mapping (both models were trained on same data)
    model_cell_size = lstm.cell_size
    print(f"\n  Using model cell_size={model_cell_size} for position mapping")

    # Load YOLO
    print("\n── Loading YOLO ──")
    yolo = YOLO(args.yolo)

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

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))

    # ── Tracking state ────────────────────────────────────────────────────
    vehicle_cells = defaultdict(list)  # {tid: [(cx, cy), ...]}
    prev_cells = {}
    prev_frame_ids = set()
    frame_number = 0

    # Predictions per vehicle
    lstm_predictions = {}   # {tid: {"label", "confidence", "probs", "steps"}}
    markov_predictions = {} # {tid: {"label", "confidence", "probs"}}

    # Stats — track agreement
    stats = {
        "total_completed": 0,
        "both_predicted": 0,
        "agreed": 0,
    }

    show_grid = False
    paused = False
    display_mode_idx = 0  # 0=both, 1=lstm_only, 2=markov_only

    print(f"\nRunning... SPACE pause | 'q' quit | 'g' grid | 'm' mode\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_frame_ids = set()

        if show_grid:
            frame = draw_grid(frame, args.cell_size)

        # Draw exit zones (use LSTM exit zones as primary — they come from the same data)
        frame = draw_exit_zones(frame, lstm.clusters, model_cell_size, alpha=0.3)

        # YOLO tracking
        results = yolo.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_cell = pixel_to_cell(cx, cy, model_cell_size)
                current_frame_ids.add(tid)

                # Record cell (only if different from last)
                if tid not in prev_cells or prev_cells[tid] != current_cell:
                    vehicle_cells[tid].append(current_cell)
                prev_cells[tid] = current_cell

                cells_seq = vehicle_cells[tid]

                # ── LSTM prediction ───────────────────────────────────
                lstm_label, lstm_conf, lstm_probs = lstm.predict(cells_seq)
                if lstm_label is not None:
                    lstm_predictions[tid] = {
                        "label": lstm_label,
                        "confidence": lstm_conf,
                        "probs": lstm_probs,
                        "steps": len(cells_seq)
                    }

                # ── Markov prediction ─────────────────────────────────
                markov_label, markov_conf, markov_probs = markov.predict(cells_seq)
                if markov_label is not None:
                    markov_predictions[tid] = {
                        "label": markov_label,
                        "confidence": markov_conf,
                        "probs": markov_probs
                    }

                # ── Draw vehicle ──────────────────────────────────────
                display_mode = DISPLAY_MODES[display_mode_idx]

                has_lstm = tid in lstm_predictions
                has_markov = tid in markov_predictions

                # Determine box color: use LSTM color if available, else Markov, else gray
                if has_lstm and display_mode != "markov_only":
                    pred = lstm_predictions[tid]
                    label_idx = lstm.label_map.get(pred["label"], 0)
                    box_color = EXIT_COLORS[label_idx % len(EXIT_COLORS)]
                elif has_markov and display_mode != "lstm_only":
                    # Map markov label to an index based on the zone list
                    zone_labels = [z["label"] for z in markov.exit_zones]
                    label_idx = zone_labels.index(markov_predictions[tid]["label"]) if markov_predictions[tid]["label"] in zone_labels else 0
                    box_color = EXIT_COLORS[label_idx % len(EXIT_COLORS)]
                else:
                    box_color = (128, 128, 128)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                # ── Text overlay ──────────────────────────────────────
                y_text = y1 - 8
                text_lines = []

                if display_mode in ("both", "lstm_only") and has_lstm:
                    pred = lstm_predictions[tid]
                    steps_text = "prior" if pred["steps"] < 3 else f"{pred['steps']}s"
                    text_lines.append(
                        (f"L: {pred['label']} {pred['confidence']:.0%} ({steps_text})",
                         (0, 255, 255))  # Cyan for LSTM
                    )

                if display_mode in ("both", "markov_only") and has_markov:
                    pred = markov_predictions[tid]
                    text_lines.append(
                        (f"M: {pred['label']} {pred['confidence']:.0%}",
                         (255, 200, 0))  # Light blue for Markov
                    )

                # Agreement indicator when both available
                if display_mode == "both" and has_lstm and has_markov:
                    agree = lstm_predictions[tid]["label"] == markov_predictions[tid]["label"]
                    symbol = "✓ AGREE" if agree else "✗ DIFFER"
                    symbol_color = (0, 255, 0) if agree else (0, 0, 255)
                    text_lines.append((symbol, symbol_color))

                if not text_lines:
                    cv2.putText(frame, f"ID:{tid} (waiting...)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (128, 128, 128), 1)

                # Draw text lines above box
                for i, (text, color) in enumerate(reversed(text_lines)):
                    y_pos = y1 - 8 - i * 16
                    cv2.putText(frame, text, (x1, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw ID at top
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 8 - len(text_lines) * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Confidence bars
                bar_y = y2 + 4
                if display_mode in ("both", "lstm_only") and has_lstm:
                    conf = lstm_predictions[tid]["confidence"]
                    bar_w = int(conf * 50)
                    cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + 5), (0, 255, 255), -1)
                    cv2.rectangle(frame, (x1, bar_y), (x1 + 50, bar_y + 5), (0, 255, 255), 1)
                    bar_y += 8

                if display_mode in ("both", "markov_only") and has_markov:
                    conf = markov_predictions[tid]["confidence"]
                    bar_w = int(conf * 50)
                    cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + 5), (255, 200, 0), -1)
                    cv2.rectangle(frame, (x1, bar_y), (x1 + 50, bar_y + 5), (255, 200, 0), 1)

                # Draw Markov predicted path (subtle)
                if display_mode in ("both", "markov_only") and has_markov:
                    path = markov.get_predicted_path(current_cell, steps=6)
                    frame = draw_markov_path(frame, path, model_cell_size, color=(255, 200, 0))

        # ── Track disappeared vehicles ────────────────────────────────
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            stats["total_completed"] += 1
            has_l = tid in lstm_predictions
            has_m = tid in markov_predictions
            if has_l and has_m:
                stats["both_predicted"] += 1
                if lstm_predictions[tid]["label"] == markov_predictions[tid]["label"]:
                    stats["agreed"] += 1
                print(f"  Vehicle {tid} exited | "
                      f"LSTM: {lstm_predictions[tid]['label']} ({lstm_predictions[tid]['confidence']:.0%}) | "
                      f"Markov: {markov_predictions[tid]['label']} ({markov_predictions[tid]['confidence']:.0%}) | "
                      f"{'AGREE' if lstm_predictions[tid]['label'] == markov_predictions[tid]['label'] else 'DIFFER'}")
            elif has_l:
                print(f"  Vehicle {tid} exited | LSTM: {lstm_predictions[tid]['label']} | Markov: N/A")
            elif has_m:
                print(f"  Vehicle {tid} exited | LSTM: N/A | Markov: {markov_predictions[tid]['label']}")

        prev_frame_ids = current_frame_ids

        # ── HUD ───────────────────────────────────────────────────────
        mode_label = DISPLAY_MODES[display_mode_idx].replace("_", " ").upper()

        # Agreement rate
        agree_pct = (stats["agreed"] / stats["both_predicted"] * 100) if stats["both_predicted"] > 0 else 0

        cv2.putText(frame, f"Frame: {frame_number}/{total_frames} | "
                    f"Active: {len(current_frame_ids)} | "
                    f"Completed: {stats['total_completed']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode_label} (m) | "
                    f"Grid: {'ON' if show_grid else 'OFF'} (g) | "
                    f"Agreement: {agree_pct:.0f}% ({stats['agreed']}/{stats['both_predicted']})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Legend
        y_off = 80
        cv2.putText(frame, "L = LSTM", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "M = Markov", (10, y_off + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        if writer:
            writer.write(frame)
            if frame_number % 30 == 0:
                print(f"  Processed frame {frame_number}/{total_frames}...")
        else:
            cv2.imshow("Dual Model Prediction (LSTM + Markov)", frame)

            wait_time = int(1000 / fps) if writer is None else 1
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                while paused:
                    k2 = cv2.waitKey(30) & 0xFF
                    if k2 == ord(' '):
                        paused = False
                        break
                    elif k2 == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif k2 == ord('g'):
                        show_grid = not show_grid
                    elif k2 == ord('m'):
                        display_mode_idx = (display_mode_idx + 1) % len(DISPLAY_MODES)
                        print(f"  Display mode: {DISPLAY_MODES[display_mode_idx]}")
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('m'):
                display_mode_idx = (display_mode_idx + 1) % len(DISPLAY_MODES)
                print(f"  Display mode: {DISPLAY_MODES[display_mode_idx]}")

    cap.release()
    if writer:
        writer.release()
    if not args.output:
        cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DUAL MODEL SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total vehicles completed:   {stats['total_completed']}")
    print(f"  Both models predicted:      {stats['both_predicted']}")
    print(f"  Models agreed:              {stats['agreed']}")
    if stats['both_predicted'] > 0:
        print(f"  Agreement rate:             {stats['agreed']/stats['both_predicted']:.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
