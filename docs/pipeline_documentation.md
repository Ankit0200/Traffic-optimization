# Vision-Based Traffic Analysis Pipeline — Complete Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 1: Vehicle Detection (YOLOv8)](#3-stage-1-vehicle-detection-yolov8)
4. [Stage 2: Grid Discretization](#4-stage-2-grid-discretization)
5. [Stage 3: Nine-Zone Transition Probability Model](#5-stage-3-nine-zone-transition-probability-model)
6. [Stage 4: Trajectory Extraction](#6-stage-4-trajectory-extraction)
7. [Stage 5: LSTM Turn Intention Predictor](#7-stage-5-lstm-turn-intention-predictor)
8. [Stage 6: Queue Manager](#8-stage-6-queue-manager)
9. [Stage 7: Arrival Rate Estimator](#9-stage-7-arrival-rate-estimator)
10. [Data Flow Summary](#10-data-flow-summary)
11. [File Reference](#11-file-reference)

---

## 1. Project Overview

This project builds a complete vision-based traffic analysis pipeline that takes raw intersection video as input and produces real-time per-vehicle turn intention predictions, per-approach queue counts, and per-approach arrival rates. Every component is learned from data — no manual lane marking, no hardcoded geometry, no intersection-specific configuration.

The pipeline is designed so that each stage feeds into the next:

```
Video → YOLO Detection → Grid Cells → Transition Model → Trajectories
                                                              ↓
                                              LSTM Training ← Exit Clustering
                                                    ↓
                                              Real-Time Prediction
                                                    ↓
                                         Queue Manager + Arrival Rates
```

### Design Principles

- **No manual configuration**: Approaches, exit zones, and queue regions are all auto-discovered from trajectory data
- **Lane-free**: Works regardless of lane markings, lane count, or intersection geometry
- **Modular**: Each component can run independently — the queue manager doesn't depend on the LSTM, the LSTM doesn't depend on the queue manager
- **Cell-based**: All spatial reasoning uses a discrete grid, not raw pixel coordinates

---

## 2. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT: Video File                            │
│                    (intersection footage, any angle)                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: YOLOv8 Vehicle Detection + ByteTrack                      │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Raw video frames                                            │
│  Output: Per-frame bounding boxes with persistent track IDs          │
│  Model:  YOLOv8n fine-tuned on VisDrone (aerial vehicle detection)   │
│  Classes: car(3), van(4), truck(5), bus(8)                           │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ (x1, y1, x2, y2, track_id) per vehicle
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Grid Discretization                                        │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Pixel coordinates (cx, cy) = center of bounding box         │
│  Output: Grid cell (col, row) and linear cell ID                     │
│  Method: cell = (cx // cell_size, cy // cell_size)                   │
│  Linear ID: id = (col+1) + k*(row), where k = frame_width/cell_size │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ cell transitions per vehicle per frame
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Nine-Zone Transition Model                                 │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Stream of (from_cell, to_cell) transitions                  │
│  Output: P(c' | c) for each cell → 8 neighbor probabilities          │
│  Also:   Raw trajectory sequences, entry/exit points                 │
│  Saves:  transitions.json + trajectories.json                        │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ trajectories.json
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Exit Zone Clustering + Labeling                            │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Trajectory endpoints                                        │
│  Output: Auto-discovered exit zones (exit_0, exit_1, ...)            │
│  Method: Greedy clustering on endpoints near frame edges             │
│  Labels: Each trajectory gets an exit label for supervised training   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ labeled trajectories + exit clusters
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5: LSTM Turn Intention Predictor                              │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Partial cell trajectory → (x, y, dx, dy) features          │
│  Output: Predicted exit zone + confidence                            │
│  Training: Data augmentation via partial trajectory slicing          │
│  Also:   Wait priors (start-cell → exit distribution)                │
│  Saves:  lstm_model.pt                                               │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ trained model
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 6: Queue Manager                                              │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Trajectory data (for approach discovery) + live detections   │
│  Output: Per-approach queue counts, vehicle-approach assignments      │
│  Method: Velocity-based stop detection, auto-discovered approaches   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ queue state per approach
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 7: Arrival Rate Estimator                                     │
│  ─────────────────────────────────────────────────────────────────── │
│  Input:  Vehicle-approach assignments from QueueManager               │
│  Output: λ_k(t) — vehicles/second per approach (rolling window)      │
│  Method: Rolling deque with configurable window duration              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Vehicle Detection (YOLOv8)

### Purpose

Detect and track individual vehicles across video frames with persistent identity. Each vehicle gets a unique track ID that persists across its entire journey through the intersection.

### Model

- **Architecture**: YOLOv8n (nano — smallest, fastest variant)
- **Training data**: VisDrone dataset (aerial/drone imagery of traffic)
- **Model file**: `models/yolov8n-visdrone.pt`
- **Why VisDrone**: Standard COCO-trained YOLO detects vehicles from street-level perspectives. VisDrone is trained on top-down/aerial views, which matches our intersection monitoring camera angle.

### Detection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `classes` | [3, 4, 5, 8] | car, van, truck, bus (VisDrone class IDs) |
| `conf` | 0.15 | Low confidence threshold to catch distant/small vehicles |
| `imgsz` | 640 | Inference resolution (px) |
| `persist` | True | Enable ByteTrack persistent tracking |

### Tracking

YOLO's built-in ByteTrack tracker assigns each detected vehicle a persistent integer ID. When a vehicle temporarily disappears behind another or is missed for a frame, ByteTrack re-associates it with the same ID. When a vehicle truly exits the frame, its ID is retired and the track is considered complete.

### Output per frame

For each detected vehicle:
- Bounding box: `(x1, y1, x2, y2)` in pixels
- Center point: `cx = (x1+x2)/2, cy = (y1+y2)/2`
- Track ID: persistent integer

---

## 4. Stage 2: Grid Discretization

### Purpose

Convert continuous pixel coordinates into discrete grid cells. This discretization is fundamental — all downstream components (transitions, trajectories, LSTM features, queue regions) operate on grid cells, not raw pixels.

### Method

The video frame is divided into a uniform grid of square cells:

```
cell_col = pixel_x // cell_size
cell_row = pixel_y // cell_size
```

### Linear Cell ID Encoding

For storage and the professor's formula compatibility, each `(col, row)` cell is encoded as a single integer:

```
id = (col + 1) + k * row
```

where `k = frame_width // cell_size` (number of cells per row).

The inverse:
```
row = (id - 1) // k
col = (id - 1) % k
```

This is implemented in `src/grid_utils.py`:
- `cell_to_id(col_0, row_0, k)` → linear ID (1-based)
- `id_to_cell(cell_id, k)` → `(col_0, row_0)` (0-based)

### Cell Size

| Cell Size | Grid Dimensions (1920x1080) | Total Cells | Use Case |
|-----------|----------------------------|-------------|----------|
| 30 px | 64 x 36 | 2,304 | Fine-grained transitions |
| 50 px | 38 x 21 | 798 | LSTM training, queue detection |

The cell size of **50 pixels** is used for the LSTM and queue manager. It provides enough spatial resolution to distinguish approach corridors and turning movements while keeping trajectory sequences short enough for efficient LSTM training.

### Example

On a 1920x1080 video with cell_size=50:
```
k = 1920 // 50 = 38 cells per row
Grid: 38 columns x 21 rows = 798 cells

Vehicle at pixel (500, 300):
  col = 500 // 50 = 10
  row = 300 // 50 = 6
  cell = (10, 6)
  linear ID = (10+1) + 38*6 = 11 + 228 = 239
```

---

## 5. Stage 3: Nine-Zone Transition Probability Model

**File**: `src/nine_zone_prob.py`

### Purpose

Build a Markov transition model that captures, for every cell in the grid, the probability of a vehicle moving to each of its 8 neighbors. This is the foundational data collection step — it produces the transition probabilities AND the raw trajectory data that all downstream components use.

### The Nine-Zone Concept

For any cell `(cx, cy)`, there are exactly 8 possible next cells (the "nine zone" includes the cell itself, though we track 8 transitions — self/stay is excluded):

```
┌──────────┬──────────┬──────────┐
│ (cx-1,   │ (cx,     │ (cx+1,   │
│  cy-1)   │  cy-1)   │  cy-1)   │
├──────────┼──────────┼──────────┤
│ (cx-1,   │  CURRENT │ (cx+1,   │
│  cy)     │ (cx,cy)  │  cy)     │
├──────────┼──────────┼──────────┤
│ (cx-1,   │ (cx,     │ (cx+1,   │
│  cy+1)   │  cy+1)   │  cy+1)   │
└──────────┴──────────┴──────────┘
```

### Transition Recording

A transition is recorded ONLY when a vehicle moves to a **different** cell. If a vehicle stays in the same cell between frames, no transition is recorded. This means transitions represent actual movement, not idle time.

### Laplace Smoothing

To prevent zero probabilities (which would make Markov predictions fail for unseen transitions), Laplace smoothing is applied:

```
P(c' | c) = (count(c → c') + 1) / (total_transitions(c) + 8)
```

This adds a pseudocount of 1 to every neighbor, ensuring no transition has probability 0.

### Clamping

If YOLO tracking jumps a vehicle by more than 1 cell between frames (due to detection gaps or fast movement), the transition is clamped to the nearest neighbor:

```python
dx = max(-1, min(1, to_cell[0] - from_cell[0]))
dy = max(-1, min(1, to_cell[1] - from_cell[1]))
clamped = (from_cell[0] + dx, from_cell[1] + dy)
```

### Visualization Features

The transition tracker provides real-time visualization:

- **Flow arrows**: Weighted average direction per cell, with color intensity proportional to traffic volume
- **Endpoint clusters**: Yellow dots showing where vehicles exit the frame
- **Predicted paths**: Green lines showing most-likely-next-cell predictions 8 steps ahead
- **Grid overlay**: Cell boundaries with linear ID labels

### Outputs

1. **Transitions JSON** (`data/transitions/<video>_transitions.json`):
   - Per-cell neighbor counts and probabilities
   - Entry points (startpoints) and exit points (endpoints)
   - Grid metadata (cell_size, grid_cols)

2. **Trajectories JSON** (`data/trajectories/<video>_trajectories.json`):
   - Per-vehicle cell sequence (unique consecutive cells only)
   - Start cell, end cell, trajectory length
   - Cells stored as linear IDs for compact storage

---

## 6. Stage 4: Trajectory Extraction and Exit Clustering

**File**: `src/lstm_predictor.py` (clustering functions)

### Purpose

Auto-discover exit zones from trajectory endpoints and label each trajectory with its exit zone for supervised LSTM training. No manual annotation required.

### Exit Zone Discovery

1. **Collect endpoints**: For each completed trajectory, take the last cell
2. **Filter**: Only keep endpoints that:
   - Are near the frame edges (within 15% of border) — real exits
   - Have sufficient displacement from start (>10 cells Euclidean) — actually traversed the intersection
3. **Cluster**: Greedy spatial clustering with Manhattan distance radius of 4 cells
4. **Label**: Each cluster becomes an exit zone (`exit_0`, `exit_1`, etc.)

### Clustering Algorithm

```python
sorted_endpoints = sorted by frequency (descending)
for each endpoint (highest count first):
    if not already assigned:
        create new cluster
        add all unassigned endpoints within radius=4 (Manhattan distance)
        if cluster has >= 3 trajectories:
            keep as exit zone
```

### Typical Result

For a standard 4-way intersection:
```
exit_0: center=(0.0, 10.7),  count=31  → West exit
exit_1: center=(37.0, 11.5), count=24  → East exit
exit_2: center=(17.5, 21.0), count=28  → South exit
exit_3: center=(19.5, 0.0),  count=26  → North exit
```

4 exit zones are discovered automatically, corresponding to the 4 departure directions.

### Labeling

Each trajectory whose endpoint falls within a cluster's cells gets that cluster's label. Trajectories with endpoints not in any cluster are discarded (unlabeled). Typical labeling rate: 70-85% of trajectories.

---

## 7. Stage 5: LSTM Turn Intention Predictor

**File**: `src/lstm_predictor.py`

### Purpose

Predict which exit zone a vehicle will take based on its **partial** trajectory — ideally within the first 3-5 cell transitions, before the vehicle reaches the intersection center.

### Feature Engineering

Each trajectory step produces 4 features:

```
For step i at cell (x, y):
  x_norm = x / 40.0     # normalized column position
  y_norm = y / 22.0     # normalized row position  
  dx = (x - x_prev) / 5.0   # normalized horizontal movement
  dy = (y - y_prev) / 5.0   # normalized vertical movement
```

First step uses `dx=0, dy=0`.

**Why these features**: Position (x, y) tells the model where the vehicle is. Direction (dx, dy) tells the model which way it's heading. Together, a vehicle at the north approach moving south-east is very different from one at the same position moving south-west — the direction disambiguates the turn intention.

### Model Architecture

```
Input: (batch, seq_len, 4)  — variable-length sequences of (x, y, dx, dy)
  │
  ▼
PackedSequence  — handles variable-length sequences efficiently
  │
  ▼
LSTM(input=4, hidden=64, layers=2, dropout=0.3)
  │
  ▼
Last hidden state: (batch, 64)
  │
  ▼
Dropout(0.3)
  │
  ▼
Linear(64 → 32) + ReLU + Dropout(0.3)
  │
  ▼
Linear(32 → num_classes)  — logits for each exit zone
  │
  ▼
Softmax → probability distribution over exit zones
```

| Parameter | Value |
|-----------|-------|
| Input size | 4 (x, y, dx, dy) |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.3 |
| FC hidden | 32 |
| Output | num_classes (typically 4 for a 4-way intersection) |
| Total parameters | ~45,000 |

### Data Augmentation

The key training technique: **partial trajectory slicing**. Each full trajectory of length N produces (N - min_steps + 1) training samples:

```
Full trajectory: [c0, c1, c2, c3, c4, c5, c6, c7]  →  exit_2

Training samples:
  [c0, c1, c2]                    →  exit_2  (3 steps)
  [c0, c1, c2, c3]               →  exit_2  (4 steps)
  [c0, c1, c2, c3, c4]           →  exit_2  (5 steps)
  [c0, c1, c2, c3, c4, c5]       →  exit_2  (6 steps)
  [c0, c1, c2, c3, c4, c5, c6]   →  exit_2  (7 steps)
  [c0, c1, c2, c3, c4, c5, c6, c7] → exit_2  (8 steps, full)
```

This teaches the model to predict correctly from **incomplete information** — the exact scenario it faces in real-time deployment.

### Wait Priors

Before the LSTM has enough trajectory steps (< 3 cells), the model falls back to **wait priors**: historical probability of taking each exit given just the starting cell.

```
P(exit | start_cell) = count(start_cell → exit) / count(start_cell → any_exit)
```

For example, if 60% of vehicles entering from cell (19, 0) historically go to exit_2 (south), the model predicts exit_2 at 60% confidence even before the vehicle moves.

### Queue Zone Discovery

The LSTM training also auto-discovers **queue zones** — regions where vehicles tend to enter and queue before proceeding. These are found by:

1. Collecting the first 1/3 of each trajectory's cells (approach corridor)
2. Clustering these cells spatially
3. Assigning each cluster to the nearest frame edge (NB, SB, EB, WB)

These queue zones are saved with the model and used by the Queue Manager.

### Training

- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss**: CrossEntropyLoss
- **Gradient clipping**: max norm 1.0
- **Early stopping**: patience=10 on validation accuracy
- **Train/val split**: 80/20, stratified by exit label
- **Batch size**: 32

### Early Prediction Accuracy

The model is evaluated on how accurately it predicts with only N steps:

```
 3 steps:  62%  ████████████████████
 4 steps:  71%  █████████████████████████
 5 steps:  78%  ███████████████████████████████
 6 steps:  82%  ██████████████████████████████████
 7 steps:  85%  ████████████████████████████████████
```

By 5 steps (~2-3 seconds of movement), the model reaches 78% accuracy — sufficient for signal control decisions.

### Model Output

Saved to `models/<video>_lstm_model.pt` containing:
- `model_state`: Trained weights
- `label_map`: `{"exit_0": 0, "exit_1": 1, ...}`
- `clusters`: Exit zone definitions (center, cells, count)
- `cell_size`: Grid cell size used during training
- `wait_priors`: Start-cell → exit probability distributions
- `queue_zones`: Auto-discovered queue regions per approach
- `model_config`: Architecture hyperparameters

---

## 8. Stage 6: Queue Manager

**File**: `src/queue_manager.py`

### Purpose

Detect which vehicles are queued (stopped) at each approach of the intersection, without requiring lane information or predefined zones. Everything is learned from trajectory data.

### Key Design Decision: Lane-Free

Traditional queue detection assigns vehicles to lanes. This requires either:
- Manual lane polygon annotation (doesn't generalize)
- Lane detection algorithms (fragile, fails on worn markings)

Our approach skips lanes entirely. Vehicles are assigned to **approaches** (spatial clusters of entry points), and "queued" is determined by **velocity**, not position.

### Approach Auto-Discovery

Approaches are discovered by clustering trajectory start points:

1. Collect the first cell of every trajectory
2. Greedy cluster with Manhattan distance radius = 8 cells
3. Filter: reject clusters in the intersection center (margin = 35%)
4. Filter: reject clusters not near a frame edge (within 20%)
5. Build ROI: for each approach, collect the first 1/4 of member trajectories' cells

This produces approach regions like:
```
A0: center=(17.3, 1.7)   → North approach (41 trajectories)
A1: center=(2.6, 11.4)   → West approach  (30 trajectories)
A2: center=(19.5, 19.3)  → South approach (26 trajectories)
A3: center=(34.8, 9.3)   → East approach  (21 trajectories)
```

### Vehicle-Approach Assignment

When a vehicle first appears:
1. Check if its cell is inside exactly one approach ROI → assign
2. If in multiple ROIs → assign to nearest approach center
3. If in no ROI → assign to nearest approach center (if within 3x radius)
4. If in the intersection center → don't assign (wait until it exits)

Once assigned, the assignment persists for the vehicle's lifetime.

### Velocity-Based Queue Detection

Each vehicle's velocity is computed from its pixel positions:

```python
# Keep last 5 pixel positions
positions = [(px1,py1), (px2,py2), ..., (px5,py5)]

# Average displacement per frame
avg_displacement = total_pixel_distance / (num_positions - 1)

# Convert to m/s
velocity = avg_displacement * fps * pixels_to_meters
```

A vehicle is **queued** if `velocity < 1.0 m/s`.

### Calibration

The `pixels_to_meters` scale is auto-calibrated by:
1. Finding the two farthest approach centers
2. Assuming they are 30 meters apart (typical intersection width)
3. Computing: `pixels_to_meters = 30.0 / pixel_distance`

### Queue State Output

```python
queue_mgr.compute_state(active_cells)
# Returns:
# {
#   "A0": {"queue": 3, "total": 5},   # 3 stopped, 5 total on North
#   "A1": {"queue": 6, "total": 8},   # 6 stopped, 8 total on West
#   "A2": {"queue": 1, "total": 2},
#   "A3": {"queue": 0, "total": 3},
# }
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VELOCITY_THRESHOLD` | 1.0 m/s | Below this, vehicle is "queued" |
| `SMOOTHING_FRAMES` | 5 | Velocity averaging window |
| `APPROACH_RADIUS` | 8 cells | Clustering radius for start points |
| `MIN_APPROACH_COUNT` | 21 | Minimum trajectories to form an approach |
| `INTERSECTION_MARGIN` | 0.35 | Center 35% excluded as intersection |
| `ASSUMED_INTERSECTION_DIST` | 30.0 m | For pixel-to-meter calibration |

---

## 9. Stage 7: Arrival Rate Estimator

**File**: `src/arrival_rate.py`

### Purpose

Measure how many new vehicles arrive at each approach per unit time. This gives `λ_k(t)` — the real-time arrival rate for approach k — which indicates whether traffic is building up, steady, or subsiding.

### Why It Matters

Queue count tells you the **current** state. Arrival rate tells you the **trend**:
- Queue = 5, arrival = 0.5 veh/s → queue is growing, will need extended green
- Queue = 5, arrival = 0.05 veh/s → queue will clear soon, current green is enough
- Queue = 0, arrival = 0.8 veh/s → surge incoming, should prepare

### Method

Uses a **rolling window deque** per approach:

```
For approach A0:
  arrivals_deque = [frame_100, frame_145, frame_210, frame_280, ...]
  
  At frame 300 with window = 15 seconds (450 frames at 30fps):
    1. Remove all entries < frame_300 - 450 = frame -150 (cutoff)
    2. Count remaining entries
    3. λ = count / 15.0 seconds
```

### Recording Arrivals

An arrival is recorded exactly once per vehicle, the first time it is assigned to an approach by the QueueManager:

```python
if not had_approach and vid in queue_mgr.vehicle_approach:
    arrival_est.record_arrival(vid, queue_mgr.vehicle_approach[vid], frame_number)
```

This ensures:
- Each vehicle is counted once (not every frame)
- The arrival is attributed to the correct approach
- Timing is accurate (recorded at first detection, not delayed)

### Output

```python
rates = arrival_est.compute_all_rates(current_frame)
# Returns:
# {
#   "A0": 0.33,   # 0.33 vehicles/second arriving at North
#   "A1": 0.13,   # 0.13 vehicles/second arriving at West
#   "A2": 0.27,
#   "A3": 0.20,
# }
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SECONDS` | 15.0 s | Rolling window duration |

### Complexity

- `record_arrival()`: O(1)
- `compute_rate()`: O(k) where k = number of expired entries to trim (typically 0-1 per call)
- Memory: O(n) where n = vehicles in the current window

### Optional CSV Logging

When `log_enabled=True`, every arrival event is written to CSV:

```csv
frame,timestamp_sec,approach,vehicle_id
100,3.33,A0,15
145,4.83,A1,22
210,7.00,A0,31
```

This enables offline analysis of arrival patterns, peak detection, and historical comparison.

---

## 10. Data Flow Summary

```
Video File (1920x1080 @ 30fps)
    │
    ├──→ YOLOv8 Detection (per frame)
    │       └── Bounding boxes + Track IDs
    │
    ├──→ Grid Discretization (cell_size=50)
    │       └── Cell (col, row) per vehicle
    │
    ├──→ Nine-Zone Transitions (per cell change)
    │       ├── transitions.json    (P(c'|c) for all cells)
    │       └── trajectories.json   (cell sequences per vehicle)
    │
    ├──→ Exit Clustering (from trajectory endpoints)
    │       └── exit_0, exit_1, exit_2, exit_3 (auto-discovered)
    │
    ├──→ LSTM Training (from labeled trajectories)
    │       └── lstm_model.pt       (turn predictor + metadata)
    │
    ├──→ Queue Manager (from trajectory start points)
    │       ├── A0, A1, A2, A3      (auto-discovered approaches)
    │       └── Queue count per approach (real-time)
    │
    └──→ Arrival Rate Estimator (from approach assignments)
            └── λ per approach      (vehicles/second, rolling window)
```

### Offline vs Real-Time

| Component | Offline (training) | Real-Time (inference) |
|-----------|-------------------|----------------------|
| YOLO | Runs on full video | Runs per frame |
| Grid | Same | Same |
| Transitions | Accumulates all data | Not used directly |
| Exit clustering | Runs once on all endpoints | Pre-computed (in model) |
| LSTM | Trains on augmented data | Predicts from partial trajectory |
| Queue Manager | Bootstraps approaches from trajectories | Updates per frame |
| Arrival Rate | Not used | Computes rolling λ per frame |

---

## 11. File Reference

### Source Code (`src/`)

| File | Purpose | Dependencies |
|------|---------|-------------|
| `grid_utils.py` | `cell_to_id()` / `id_to_cell()` conversions | None |
| `nine_zone_prob.py` | Transition tracking + trajectory extraction | grid_utils, YOLO, OpenCV |
| `lstm_predictor.py` | LSTM training + exit clustering + queue zone discovery | grid_utils, PyTorch, scikit-learn |
| `test_lstm_realtime.py` | Real-time LSTM prediction on video | lstm_predictor, queue_manager, YOLO |
| `queue_manager.py` | Lane-free queue detection per approach | grid_utils, NumPy |
| `arrival_rate.py` | Rolling-window arrival rate estimator | None (stdlib only) |
| `testing_queue.py` | Queue-only testing/debugging tool | queue_manager, arrival_rate, YOLO |

### Data (`data/`)

| Path | Contents |
|------|----------|
| `data/transitions/` | Per-cell transition probabilities (JSON) |
| `data/trajectories/` | Per-vehicle cell sequences (JSON) |

### Models (`models/`)

| File | Contents |
|------|----------|
| `yolov8n-visdrone.pt` | YOLOv8 nano fine-tuned on VisDrone |
| `intersection_sim_trajectories_lstm_model.pt` | Trained LSTM + exit zones + wait priors + queue zones |

### Running the Pipeline

```bash
# Step 1: Track vehicles and build transitions + trajectories
cd src/
python nine_zone_prob.py --video ../data/sumo_simulation/intersection_sim.mp4 --cell_size 50

# Step 2: Train LSTM on extracted trajectories
python lstm_predictor.py --data ../data/trajectories/intersection_sim_trajectories.json

# Step 3: Run real-time prediction with queue detection
python test_lstm_realtime.py \
    --video ../data/sumo_simulation/intersection_sim.mp4 \
    --model ../models/intersection_sim_trajectories_lstm_model.pt \
    --trajectories ../data/trajectories/intersection_sim_trajectories.json

# Step 4: Run queue-only testing with arrival rates
python testing_queue.py \
    --video ../data/sumo_simulation/intersection_sim.mp4 \
    --model ../models/intersection_sim_trajectories_lstm_model.pt \
    --yolo ../models/yolov8n-visdrone.pt \
    --trajectories ../data/trajectories/intersection_sim_trajectories.json
```
