"""
Real-Time LSTM Turn Prediction
================================
Usage:
    python test_lstm_realtime.py --video control.mp4 --model control_trajectories_lstm_model.pt

Runs YOLO tracking on video and uses the trained LSTM to predict
each vehicle's exit direction in real-time as it moves.

Controls:
    SPACE : Pause / Resume
    'q'   : Quit
    'g'   : Toggle grid
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


def draw_exit_zones(frame, clusters, label_map, display_cell_size, model_cell_size, alpha=0.45):
    """Draw highlighted colored rectangles on exit zone cells with borders."""
    overlay = frame.copy()
    
    # Scale factor from training cell size to display cell size
    scale = model_cell_size / display_cell_size if display_cell_size > 0 else 1.0

    for cl in clusters:
        label = cl["label"]
        idx = label_map.get(label, 0)
        color = EXIT_COLORS[idx % len(EXIT_COLORS)]
        
        # Scale cells
        cells = cl["cells"]

        for c in cells:
            # The cell coordinate in the original training grid
            cx_orig, cy_orig = c[0], c[1]
            
            # Map it to pixel coordinates using the original cell_size it was trained on
            px1 = int(cx_orig * model_cell_size)
            py1 = int(cy_orig * model_cell_size)
            px2 = px1 + model_cell_size
            py2 = py1 + model_cell_size
            
            # Filled rectangle on overlay
            cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
            # Thick border directly on frame so it's always fully visible
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        # Label with dark background for readability
        center_x = int(cl["center"][0] * model_cell_size + model_cell_size // 2)
        center_y = int(cl["center"][1] * model_cell_size + model_cell_size // 2)
        text = label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(overlay, (center_x - 22, center_y - th - 4),
                      (center_x + tw - 16, center_y + 4), (0, 0, 0), -1)
        cv2.putText(overlay, text, (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTOR CLASS — wraps the model for easy real-time use
# ═══════════════════════════════════════════════════════════════════════════

class RealtimePredictor:
    def __init__(self, model_path, device='cpu'):
        # Load saved model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.label_map = checkpoint["label_map"]
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.clusters = checkpoint["clusters"]
        self.cell_size = checkpoint["cell_size"]
        config = checkpoint["model_config"]

        # Build and load model
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

        self.min_steps = 3  # Minimum steps before predicting

        print(f"Model loaded: {len(self.label_map)} exit classes")
        for cl in self.clusters:
            print(f"  {cl['label']}: center={cl['center']}, count={cl['count']}")

    def predict(self, cell_sequence):
        """
        Predict exit from a partial cell sequence.

        Args:
            cell_sequence: list of (cx, cy) tuples

        Returns:
            (predicted_label, confidence, all_probabilities)
            or (None, 0, {}) if not enough data yet
        """
        if len(cell_sequence) < self.min_steps:
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
# MAIN — Run on video
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real-Time LSTM Turn Prediction")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to trained LSTM model (.pt)")
    parser.add_argument("--yolo", default="../models/10_epoch.pt", help="YOLO model path")
    parser.add_argument("--cell_size", type=int, default=50, help="Grid cell size")
    parser.add_argument("--output", help="Path to save output video (e.g., output.mp4)")
    args = parser.parse_args()

    # Load predictor
    print("\n── Loading LSTM model ──")
    predictor = RealtimePredictor(args.model)

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

    # Tracking state
    display_cell_size = args.cell_size
    model_cell_size = predictor.cell_size
    if display_cell_size != model_cell_size:
        print(f"Warning: Display cell size ({display_cell_size}) differs from model's cell size ({model_cell_size}). "
              f"Predictions will use the model's scale.")

    # {track_id: [(cx, cy), ...]} — unique cells per vehicle
    vehicle_cells = defaultdict(list)
    prev_cells = {}
    prev_frame_ids = set()
    frame_number = 0

    # Prediction results per vehicle
    # {track_id: {"label": str, "confidence": float, "probs": dict}}
    predictions = {}

    # Stats
    stats = {"total_predicted": 0, "correct": 0}

    show_grid = False
    paused = False

    print(f"\nRunning... SPACE pause | 'q' quit | 'g' grid\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_frame_ids = set()

        if show_grid:
            frame = draw_grid(frame, display_cell_size)

        # Draw exit zones on every frame
        frame = draw_exit_zones(frame, predictor.clusters, predictor.label_map, display_cell_size, model_cell_size)

        # YOLO tracking
        results = yolo.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # IMPORTANT: Map the vehicle's position into the CELL GRID the model was built for
                current_cell = pixel_to_cell(cx, cy, model_cell_size)
                current_frame_ids.add(tid)

                # Record cell (only if different from last)
                if tid not in prev_cells or prev_cells[tid] != current_cell:
                    vehicle_cells[tid].append(current_cell)
                prev_cells[tid] = current_cell

                # Run LSTM prediction
                cells_seq = vehicle_cells[tid]
                pred_label, confidence, all_probs = predictor.predict(cells_seq)

                if pred_label is not None:
                    predictions[tid] = {
                        "label": pred_label,
                        "confidence": confidence,
                        "probs": all_probs,
                        "steps": len(cells_seq)
                    }

                # ── Draw vehicle ──────────────────────────────────────
                # Color by prediction
                if tid in predictions:
                    pred = predictions[tid]
                    label_idx = predictor.label_map.get(pred["label"], 0)
                    color = EXIT_COLORS[label_idx % len(EXIT_COLORS)]
                    conf = pred["confidence"]

                    # Bounding box colored by predicted exit
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Prediction text
                    cv2.putText(frame, f"ID:{tid} → {pred['label']}",
                                (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                    cv2.putText(frame, f"{conf:.0%} ({pred['steps']} steps)",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (200, 200, 200), 1)

                    # Confidence bar
                    bar_w = int(conf * 60)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 12), color, -1)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + 60, y2 + 12), color, 1)
                else:
                    # Not enough data yet
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, f"ID:{tid} (waiting...)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (128, 128, 128), 1)

                # Center dot
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Detect disappeared tracks — log final prediction
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            if tid in predictions:
                pred = predictions[tid]
                stats["total_predicted"] += 1
                print(f"  Vehicle {tid} exited → predicted: {pred['label']} "
                      f"({pred['confidence']:.0%}, {pred['steps']} steps)")

        prev_frame_ids = current_frame_ids

        # ── HUD ───────────────────────────────────────────────────────
        # Legend
        y_offset = 80
        for label, idx in predictor.label_map.items():
            color = EXIT_COLORS[idx % len(EXIT_COLORS)]
            cv2.rectangle(frame, (10, y_offset), (25, y_offset + 12), color, -1)
            cv2.putText(frame, label, (30, y_offset + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18

        # Active predictions count
        active_preds = sum(1 for tid in current_frame_ids if tid in predictions)
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames} | "
                    f"Active: {len(current_frame_ids)} | "
                    f"Predicting: {active_preds} | "
                    f"Completed: {stats['total_predicted']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Grid: {'ON' if show_grid else 'OFF'} (g) | "
                    f"{'PAUSED' if paused else 'PLAYING'} (space)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if writer:
            writer.write(frame)
            if frame_number % 30 == 0:
                print(f"  Processed frame {frame_number}/{total_frames}...")
        else:
            cv2.imshow("LSTM Turn Prediction", frame)

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
            elif key == ord('g'):
                show_grid = not show_grid

    cap.release()
    if writer:
        writer.release()
    if not args.output:
        cv2.destroyAllWindows()

    # Final summary
    print(f"\n{'='*50}")
    print(f"  SESSION SUMMARY")
    print(f"  Total vehicles predicted: {stats['total_predicted']}")
    print(f"  Total vehicles tracked: {len(vehicle_cells)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()