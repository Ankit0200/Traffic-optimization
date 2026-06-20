"""
LSTM Model Evaluation on Video
================================
Usage:
    python evaluate_lstm.py --video control.mp4 --model control_trajectories_lstm_model.pt

Runs the LSTM on video, waits for each vehicle to FINISH its trajectory,
then checks if the prediction was correct by comparing the predicted exit
to the actual endpoint cluster.

Gives you:
    - Overall accuracy
    - Accuracy per exit class
    - Accuracy vs number of steps observed
    - Early prediction analysis (how soon can we predict correctly?)
    - Confidence calibration (when model says 80%, is it right 80% of the time?)
    - Per-vehicle log for debugging
"""

import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO
from lstm_predictor import TurnPredictor, trajectory_to_features


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_cell(x, y, cell_size):
    return (int(x // cell_size), int(y // cell_size))


class Predictor:
    """Wraps the trained LSTM for prediction."""
    def __init__(self, model_path, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.label_map = checkpoint["label_map"]
        self.inv_map = {v: k for k, v in self.label_map.items()}
        self.clusters = checkpoint["clusters"]
        self.cell_size = checkpoint["cell_size"]
        config = checkpoint["model_config"]

        self.model = TurnPredictor(
            input_size=config["input_size"], hidden_size=config["hidden_size"],
            num_layers=config["num_layers"], num_classes=config["num_classes"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.device = device
        self.model.to(device)

        # Build cell → label lookup for ground truth
        self.cell_to_label = {}
        for cl in self.clusters:
            for c in cl["cells"]:
                self.cell_to_label[tuple(c)] = cl["label"]

    def predict(self, cell_sequence):
        if len(cell_sequence) < 3:
            return None, 0.0, {}
        features = trajectory_to_features(cell_sequence)
        seq = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        length = torch.tensor([len(cell_sequence)], dtype=torch.long)
        with torch.no_grad():
            output = self.model(seq, length)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_idx = np.argmax(probs)
        return self.inv_map[pred_idx], float(probs[pred_idx]), \
               {self.inv_map[i]: float(p) for i, p in enumerate(probs)}

    def get_ground_truth(self, end_cell):
        """Look up which exit cluster this endpoint belongs to."""
        return self.cell_to_label.get(tuple(end_cell), None)


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM on Video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--yolo", default="../models/10_epoch.pt")
    parser.add_argument("--cell_size", type=int, default=50)
    parser.add_argument("--save_log", action="store_true", help="Save per-vehicle log to JSON")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = Predictor(args.model, device)
    yolo = YOLO(args.yolo)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cell_size = args.cell_size

    # Tracking state
    vehicle_cells = defaultdict(list)
    prev_cells = {}
    prev_frame_ids = set()
    frame_number = 0

    # Store predictions made at EVERY step for each vehicle
    # {tid: [(n_steps, predicted_label, confidence), ...]}
    prediction_history = defaultdict(list)

    # Final results when vehicle exits
    # [{tid, true_label, final_pred, final_conf, steps, correct, early_predictions}, ...]
    results = []

    print(f"\nRunning evaluation on {total_frames} frames...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        current_frame_ids = set()
        results_yolo = yolo.track(frame, persist=True, classes=[0])

        if results_yolo[0].boxes.id is not None:
            boxes = results_yolo[0].boxes.xyxy.cpu()
            track_ids = results_yolo[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                current_cell = pixel_to_cell(cx, cy, cell_size)
                current_frame_ids.add(tid)

                if tid not in prev_cells or prev_cells[tid] != current_cell:
                    vehicle_cells[tid].append(current_cell)
                prev_cells[tid] = current_cell

                # Record prediction at this step
                cells_seq = vehicle_cells[tid]
                pred_label, conf, all_probs = predictor.predict(cells_seq)
                if pred_label is not None:
                    prediction_history[tid].append({
                        "steps": len(cells_seq),
                        "prediction": pred_label,
                        "confidence": conf
                    })

        # Check disappeared vehicles — evaluate their predictions
        disappeared = prev_frame_ids - current_frame_ids
        for tid in disappeared:
            if tid not in vehicle_cells or len(vehicle_cells[tid]) < 3:
                continue

            cells_seq = vehicle_cells[tid]
            end_cell = cells_seq[-1]
            true_label = predictor.get_ground_truth(end_cell)

            if true_label is None:
                continue  # Endpoint not in any cluster — skip

            # Get final prediction
            final_pred, final_conf, _ = predictor.predict(cells_seq)

            # Check at what step the model first predicted correctly
            first_correct_step = None
            for ph in prediction_history[tid]:
                if ph["prediction"] == true_label:
                    first_correct_step = ph["steps"]
                    break

            result = {
                "track_id": tid,
                "true_label": true_label,
                "final_prediction": final_pred,
                "final_confidence": round(final_conf, 3),
                "total_steps": len(cells_seq),
                "correct": final_pred == true_label,
                "first_correct_step": first_correct_step,
                "prediction_history": prediction_history[tid]
            }
            results.append(result)

            status = "✓" if result["correct"] else "✗"
            print(f"  {status} Vehicle {tid}: true={true_label}, "
                  f"pred={final_pred} ({final_conf:.0%}), "
                  f"{len(cells_seq)} steps, "
                  f"first correct at step {first_correct_step}")

        prev_frame_ids = current_frame_ids

        if frame_number % 100 == 0:
            print(f"  ... frame {frame_number}/{total_frames}")

    cap.release()

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════

    if not results:
        print("\nNo completed vehicles matched exit clusters. Check your data.")
        return

    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*60}")

    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n  Overall accuracy: {correct}/{total} = {correct/total:.1%}")

    # Per-class accuracy
    print(f"\n  Per-class results:")
    class_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        class_results[r["true_label"]]["total"] += 1
        if r["correct"]:
            class_results[r["true_label"]]["correct"] += 1

    for label in sorted(class_results.keys()):
        cr = class_results[label]
        acc = cr["correct"] / cr["total"] if cr["total"] > 0 else 0
        bar = "█" * int(acc * 20)
        print(f"    {label}: {cr['correct']:>3}/{cr['total']:>3} = {acc:>5.1%}  {bar}")

    # Accuracy vs steps observed (at prediction time)
    print(f"\n  Accuracy by trajectory length:")
    length_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        bucket = min(r["total_steps"], 20)
        length_results[bucket]["total"] += 1
        if r["correct"]:
            length_results[bucket]["correct"] += 1

    for n in sorted(length_results.keys()):
        lr = length_results[n]
        acc = lr["correct"] / lr["total"] if lr["total"] > 0 else 0
        bar = "█" * int(acc * 20)
        print(f"    {n:>2} steps: {lr['correct']:>3}/{lr['total']:>3} = {acc:>5.1%}  {bar}")

    # Early prediction: how soon does the model get it right?
    print(f"\n  Early prediction (first correct step):")
    early_steps = [r["first_correct_step"] for r in results if r["first_correct_step"] is not None]
    if early_steps:
        print(f"    Average first correct step: {np.mean(early_steps):.1f}")
        print(f"    Median first correct step:  {np.median(early_steps):.1f}")
        print(f"    Min: {min(early_steps)}, Max: {max(early_steps)}")
        # Distribution
        step_counts = defaultdict(int)
        for s in early_steps:
            step_counts[s] += 1
        print(f"\n    Distribution:")
        for s in sorted(step_counts.keys()):
            bar = "█" * step_counts[s]
            print(f"      Step {s:>2}: {step_counts[s]:>3} vehicles  {bar}")

    never_correct = sum(1 for r in results if r["first_correct_step"] is None)
    if never_correct > 0:
        print(f"\n    Never predicted correctly: {never_correct} vehicles")

    # Confidence calibration
    print(f"\n  Confidence calibration:")
    conf_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        bucket = round(r["final_confidence"], 1)  # Round to nearest 0.1
        conf_buckets[bucket]["total"] += 1
        if r["correct"]:
            conf_buckets[bucket]["correct"] += 1

    for conf in sorted(conf_buckets.keys()):
        cb = conf_buckets[conf]
        actual_acc = cb["correct"] / cb["total"] if cb["total"] > 0 else 0
        print(f"    Conf {conf:.0%}: {cb['correct']:>3}/{cb['total']:>3} actually correct = {actual_acc:.0%}")

    # Wrong predictions detail
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  Wrong predictions ({len(wrong)}):")
        for r in wrong:
            print(f"    Vehicle {r['track_id']}: true={r['true_label']}, "
                  f"pred={r['final_prediction']} ({r['final_confidence']:.0%}), "
                  f"{r['total_steps']} steps")

    # Save log
    if args.save_log:
        log_path = Path(args.video).stem + "_eval_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                "overall_accuracy": correct / total,
                "total_evaluated": total,
                "per_class": {k: v for k, v in class_results.items()},
                "results": results
            }, f, indent=2, default=str)
        print(f"\n  Detailed log saved to: {log_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()