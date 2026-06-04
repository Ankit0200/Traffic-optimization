"""
LSTM Turn Intention Predictor
==============================
Usage:
    python lstm_predictor.py --data control_trajectories.json

Trains an LSTM to predict vehicle exit direction from partial trajectories.
Auto-labels training data by clustering endpoints — no manual exit zones needed.

Requirements:
    pip install torch numpy scikit-learn
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from grid_utils import id_to_cell

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load trajectories and auto-label via endpoint clustering
# ═══════════════════════════════════════════════════════════════════════════

def load_trajectories(filepath):
    """Load trajectory JSON and return (cell_size, trajectories).

    Supports both formats:
      - New format: cells stored as linear integer IDs (requires grid_cols in JSON)
      - Old format: cells stored as [col, row] lists

    In both cases, returns cells as (col, row) tuples so all downstream
    code (clustering, feature engineering, training) is unchanged.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    k = data.get("grid_cols")   # cells per row; present only in new-format files
    print(f"Loaded {data['total_tracks']} trajectories, "
          f"cell_size={data['cell_size']}, "
          f"grid_cols={k if k else 'N/A (old format)'}")

    trajectories = {}
    for tid, traj in data["trajectories"].items():
        cells_raw = traj["cells"]
        # Detect format from first element
        if cells_raw and isinstance(cells_raw[0], int):
            # New format: list of linear IDs  →  convert to (col, row) tuples
            cell_tuples = [id_to_cell(c, k) for c in cells_raw]
        else:
            # Old format: list of [col, row] pairs  →  just convert to tuples
            cell_tuples = [tuple(c) for c in cells_raw]

        trajectories[tid] = {
            "cells": cell_tuples,
            "start": list(cell_tuples[0]) if cell_tuples else traj.get("start"),
            "end":   list(cell_tuples[-1]) if cell_tuples else traj.get("end"),
            "length": traj["length"]
        }

    return data["cell_size"], trajectories


def cluster_endpoints(trajectories, radius=4):
    """Cluster trajectory endpoints to auto-discover exit zones."""
    ep_counts = defaultdict(list)
    
    # Calculate grid bounds for edge filtering
    if not trajectories:
        return []
    max_x = max(max(c[0] for c in traj["cells"]) for traj in trajectories.values())
    max_y = max(max(c[1] for c in traj["cells"]) for traj in trajectories.values())
    
    edge_margin_x = max_x * 0.15
    edge_margin_y = max_y * 0.15
    
    for tid, traj in trajectories.items():
        start, end = traj["start"], traj["end"]
        
        # 1. Displacement filter (must move at least 10 Euclidean cells)
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        if dist < 10:
            continue
            
        # 2. Edge filter (must end near the edges of the frame)
        is_near_edge_x = end[0] <= edge_margin_x or end[0] >= (max_x - edge_margin_x)
        is_near_edge_y = end[1] <= edge_margin_y or end[1] >= (max_y - edge_margin_y)
        
        if is_near_edge_x or is_near_edge_y:
            end_tuple = tuple(end)
            ep_counts[end_tuple].append(tid)

    # Sort by frequency
    sorted_eps = sorted(ep_counts.items(), key=lambda x: len(x[1]), reverse=True)

    clusters = []
    assigned = set()

    for cell, tids in sorted_eps:
        if cell in assigned:
            continue

        cluster_cells = set()
        cluster_tids = []

        for other_cell, other_tids in sorted_eps:
            if other_cell in assigned:
                continue
            dist = abs(cell[0] - other_cell[0]) + abs(cell[1] - other_cell[1])
            if dist <= radius:
                cluster_cells.add(other_cell)
                cluster_tids.extend(other_tids)
                assigned.add(other_cell)

        if len(cluster_tids) >= 3:
            cx = np.mean([c[0] for c in cluster_cells])
            cy = np.mean([c[1] for c in cluster_cells])
            clusters.append({
                "label": f"exit_{len(clusters)}",
                "center": (round(cx, 1), round(cy, 1)),
                "cells": cluster_cells,
                "track_ids": cluster_tids,
                "count": len(cluster_tids)
            })

    return clusters


def label_trajectories(trajectories, clusters):
    """Assign exit labels to trajectories based on endpoint clusters."""
    # Build lookup: cell → label
    cell_to_label = {}
    for cl in clusters:
        for c in cl["cells"]:
            cell_to_label[c] = cl["label"]

    labeled = []
    unlabeled_count = 0

    for tid, traj in trajectories.items():
        end = tuple(traj["end"])
        if end in cell_to_label:
            labeled.append({
                "id": tid,
                "cells": [tuple(c) for c in traj["cells"]],
                "label": cell_to_label[end]
            })
        else:
            unlabeled_count += 1

    print(f"Labeled: {len(labeled)}, Unlabeled: {unlabeled_count}")
    return labeled


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def trajectory_to_features(cells):
    """
    Convert a trajectory of cells into a feature sequence.

    For each step, compute:
        - x, y (normalized cell coordinates)
        - dx, dy (movement direction from previous step)

    Returns: numpy array of shape (seq_len, 4)
    """
    features = []
    for i, (x, y) in enumerate(cells):
        if i == 0:
            dx, dy = 0.0, 0.0
        else:
            dx = x - cells[i-1][0]
            dy = y - cells[i-1][1]

        features.append([
            x / 40.0,   # normalize x (assuming max ~38 cells wide)
            y / 22.0,   # normalize y (assuming max ~21 cells tall)
            dx / 5.0,   # normalize dx
            dy / 5.0    # normalize dy
        ])

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, labels, label_map):
        self.sequences = sequences   # list of numpy arrays (seq_len, 4)
        self.labels = labels         # list of label strings
        self.label_map = label_map   # {"exit_0": 0, "exit_1": 1, ...}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = self.label_map[self.labels[idx]]
        return seq, label, len(self.sequences[idx])


def collate_fn(batch):
    """Pad sequences to same length for batching."""
    seqs, labels, lengths = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: LSTM Model
# ═══════════════════════════════════════════════════════════════════════════

class TurnPredictor(nn.Module):
    """
    LSTM that reads a partial trajectory and predicts the exit zone.

    Input: sequence of (x, y, dx, dy) per step
    Output: probability distribution over exit zones
    """

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
        # Pack padded sequences for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Training with data augmentation
# ═══════════════════════════════════════════════════════════════════════════

def compute_wait_priors(labeled_data):
    """
    Calculate the historical probability of taking each exit given a start cell.
    Returns: { (start_x, start_y): { "exit_0": prob, ... }, ... }
    """
    start_counts = defaultdict(lambda: defaultdict(int))
    for item in labeled_data:
        start_cell = tuple(item["cells"][0])
        label = item["label"]
        start_counts[start_cell][label] += 1
        
    wait_priors = {}
    for cell, counts in start_counts.items():
        total = sum(counts.values())
        wait_priors[cell] = {label: count / total for label, count in counts.items()}
    
    print(f"Computed wait priors for {len(wait_priors)} distinct start cells.")
    return wait_priors


def discover_queue_zones(trajectories, radius=4, min_count=3):
    """
    Auto-discover queue zones from trajectory entry regions.

    Collects the first N cells of every trajectory (where vehicles enter the
    scene from an approach edge) and clusters them spatially. Each cluster
    near an edge becomes that approach's queue zone.

    Also includes any cell that appears in the first-third of multiple
    trajectories — these are the approach corridors where vehicles queue.

    Returns: {approach: set of (cx, cy) cells}
    """
    if not trajectories:
        return {}

    all_cells = [c for traj in trajectories.values() for c in traj["cells"]]
    grid_w = max(c[0] for c in all_cells) + 1
    grid_h = max(c[1] for c in all_cells) + 1

    # Collect entry region cells: first 1/3 of each trajectory (approach corridor)
    entry_counts = defaultdict(int)
    for traj in trajectories.values():
        cells = traj["cells"]
        entry_len = max(1, len(cells) // 3)
        for c in cells[:entry_len]:
            entry_counts[c] += 1

    if not entry_counts:
        print("  No entry cells found — queue zones not discovered.")
        return {}

    print(f"  Entry region cells: {len(entry_counts)} distinct cells across all trajectories.")

    # Only keep cells that appear in multiple trajectories (shared approach corridor)
    frequent = {c: n for c, n in entry_counts.items() if n >= 2}
    if not frequent:
        frequent = entry_counts  # fallback: use all

    # Cluster by proximity
    sorted_cells = sorted(frequent.items(), key=lambda x: x[1], reverse=True)
    assigned = set()
    raw_clusters = []

    for cell, count in sorted_cells:
        if cell in assigned:
            continue
        cluster_cells = {}
        for other_cell, other_count in sorted_cells:
            if other_cell in assigned:
                continue
            if abs(cell[0] - other_cell[0]) + abs(cell[1] - other_cell[1]) <= radius:
                cluster_cells[other_cell] = other_count
                assigned.add(other_cell)
        total = sum(cluster_cells.values())
        if total >= min_count:
            cx = np.mean([c[0] for c in cluster_cells])
            cy = np.mean([c[1] for c in cluster_cells])
            raw_clusters.append({
                "center": (cx, cy),
                "cells": set(cluster_cells.keys()),
                "count": total
            })

    # Assign each cluster to nearest approach edge; skip central clusters
    edge_limit = min(grid_w, grid_h) * 0.45
    queue_zones = defaultdict(set)

    for cluster in raw_clusters:
        cx, cy = cluster["center"]
        dists = {"SB": cy, "NB": grid_h - cy, "EB": cx, "WB": grid_w - cx}
        nearest = min(dists, key=dists.get)
        if dists[nearest] > edge_limit:
            continue  # too central — skip
        queue_zones[nearest].update(cluster["cells"])

    result = dict(queue_zones)
    if result:
        for approach, cells in sorted(result.items()):
            print(f"  Queue zone {approach}: {len(cells)} cells, sample={sorted(cells)[:3]}")
    else:
        print("  No queue zones discovered (all clusters too central).")
    return result


def augment_partial_trajectories(labeled_data, min_steps=3):
    """
    Data augmentation: for each full trajectory, create partial versions.
    A trajectory of length 10 produces samples of length 3, 4, 5, ..., 10.
    This teaches the model to predict from PARTIAL information.
    """
    augmented_seqs = []
    augmented_labels = []

    for item in labeled_data:
        cells = item["cells"]
        label = item["label"]

        # Create partial trajectories
        for end_idx in range(min_steps, len(cells) + 1):
            partial = cells[:end_idx]
            features = trajectory_to_features(partial)
            augmented_seqs.append(features)
            augmented_labels.append(label)

    print(f"Augmented: {len(labeled_data)} full trajectories → {len(augmented_seqs)} samples")
    return augmented_seqs, augmented_labels


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the LSTM with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for seqs, labels, lengths in train_loader:
            seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)

            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for seqs, labels, lengths in val_loader:
                seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
                outputs = model(seqs, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_loss / max(val_total, 1))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}: "
                  f"Train acc={train_acc:.3f} loss={train_loss/train_total:.4f} | "
                  f"Val acc={val_acc:.3f} loss={val_loss/max(val_total,1):.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"\n  Best validation accuracy: {best_val_acc:.3f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Evaluation — test early prediction accuracy
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_early_prediction(model, labeled_data, label_map, device='cpu'):
    """
    Test how accurately the model predicts with only the first N steps.
    This is the key metric — can we predict the turn EARLY?
    """
    model.eval()
    inv_map = {v: k for k, v in label_map.items()}
    results_by_steps = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for item in labeled_data:
            cells = item["cells"]
            true_label = item["label"]
            true_idx = label_map[true_label]

            # Test at each partial length
            for n_steps in range(3, min(len(cells) + 1, 15)):
                partial = cells[:n_steps]
                features = trajectory_to_features(partial)
                seq = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                length = torch.tensor([n_steps], dtype=torch.long)

                output = model(seq, length)
                _, predicted = output.max(1)

                results_by_steps[n_steps]["total"] += 1
                if predicted.item() == true_idx:
                    results_by_steps[n_steps]["correct"] += 1

    print(f"\n{'='*50}")
    print(f"  EARLY PREDICTION ACCURACY")
    print(f"  (How well can we predict with only N steps?)")
    print(f"{'='*50}")
    for n in sorted(results_by_steps.keys()):
        r = results_by_steps[n]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        bar = "█" * int(acc * 30)
        print(f"  {n:>2} steps: {acc:>6.1%} ({r['correct']:>3}/{r['total']:>3}) {bar}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Save model for real-time use
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model, label_map, clusters, cell_size, wait_priors, queue_zones, filepath):
    """Save trained model and metadata."""
    wait_priors_serializable = [{"cell": list(k), "probs": v} for k, v in wait_priors.items()]
    queue_zones_serializable = {
        approach: [list(c) for c in cells]
        for approach, cells in queue_zones.items()
    }

    torch.save({
        "model_state": model.state_dict(),
        "label_map": label_map,
        "clusters": [
            {"label": c["label"], "center": list(c["center"]),
             "cells": [list(cell) for cell in c["cells"]], "count": c["count"]}
            for c in clusters
        ],
        "cell_size": cell_size,
        "wait_priors": wait_priors_serializable,
        "queue_zones": queue_zones_serializable,
        "model_config": {
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": len(label_map)
        }
    }, filepath)
    print(f"  Model saved to: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LSTM Turn Intention Predictor")
    parser.add_argument("--data", required=True, help="Path to trajectories JSON")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--cluster_radius", type=int, default=4, help="Endpoint clustering radius")
    parser.add_argument("--min_traj_length", type=int, default=5, help="Minimum trajectory length to use")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Step 1: Load and label
    print("\n── Step 1: Loading trajectories ──")
    cell_size, trajectories = load_trajectories(args.data)

    # Filter short trajectories
    trajectories = {k: v for k, v in trajectories.items() if v["length"] >= args.min_traj_length}
    print(f"After filtering (min length {args.min_traj_length}): {len(trajectories)} trajectories")

    # Step 2: Auto-discover exits
    print("\n── Step 2: Clustering endpoints ──")
    clusters = cluster_endpoints(trajectories, radius=args.cluster_radius)
    print(f"Found {len(clusters)} exit zones:")
    for cl in clusters:
        print(f"  {cl['label']}: center={cl['center']}, count={cl['count']}, "
              f"cells={sorted(cl['cells'])}")

    # Step 3: Label trajectories
    print("\n── Step 3: Labeling trajectories ──")
    labeled = label_trajectories(trajectories, clusters)

    if len(labeled) < 10:
        print("ERROR: Not enough labeled trajectories. Try lower cluster_radius or more data.")
        return

    # Check class distribution
    label_counts = defaultdict(int)
    for item in labeled:
        label_counts[item["label"]] += 1
    print(f"Class distribution: {dict(label_counts)}")

    # Create label map
    labels_list = sorted(label_counts.keys())
    label_map = {label: idx for idx, label in enumerate(labels_list)}
    print(f"Label map: {label_map}")

    # Step 3.5: Compute wait priors
    print("\n── Step 3.5: Computing wait priors ──")
    wait_priors = compute_wait_priors(labeled)

    # Step 3.6: Auto-discover queue zones from stop events
    print("\n── Step 3.6: Discovering queue zones ──")
    queue_zones = discover_queue_zones(trajectories)

    # Step 4: Augment with partial trajectories
    print("\n── Step 4: Augmenting data ──")
    aug_seqs, aug_labels = augment_partial_trajectories(labeled, min_steps=3)

    # Step 5: Train/val split
    print("\n── Step 5: Training LSTM ──")
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        aug_seqs, aug_labels, test_size=0.2, random_state=42, stratify=aug_labels
    )

    train_ds = TrajectoryDataset(train_seqs, train_labels, label_map)
    val_ds = TrajectoryDataset(val_seqs, val_labels, label_map)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Build model
    model = TurnPredictor(
        input_size=4, hidden_size=64, num_layers=2,
        num_classes=len(label_map), dropout=0.3
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model = train_model(model, train_loader, val_loader, epochs=args.epochs, device=device)

    # Step 6: Evaluate early prediction
    print("\n── Step 6: Early prediction evaluation ──")
    evaluate_early_prediction(model, labeled, label_map, device=device)

    # Full evaluation on validation set
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for seqs, labels, lengths in val_loader:
            seqs, lengths = seqs.to(device), lengths.to(device)
            outputs = model(seqs, lengths)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in range(len(label_map))]
    print(f"\n── Classification Report ──")
    print(classification_report(all_true, all_preds, target_names=target_names))

    # Step 7: Save
    out_dir_lstm = PROJECT_ROOT / "models"
    out_dir_lstm.mkdir(parents=True, exist_ok=True)
    model_path = out_dir_lstm / (Path(args.data).stem + "_lstm_model.pt")
    save_model(model, label_map, clusters, cell_size, wait_priors, queue_zones, str(model_path))

    print(f"\n  To use in real-time:")
    print(f"    1. Load model from '{model_path}'")
    print(f"    2. As a vehicle moves, collect its cell sequence")
    print(f"    3. Feed partial sequence to model → get exit prediction")
    print(f"    4. Confidence increases as more steps are observed\n")


if __name__ == "__main__":
    main()