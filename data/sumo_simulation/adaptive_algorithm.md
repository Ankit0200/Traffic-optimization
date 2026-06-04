# Adaptive Signal Control Algorithm

## Overview

The adaptive controller uses three data sources to allocate green time:

1. **QueueManager** — velocity-based queue detection per approach (from `queue_manager.py`)
2. **ArrivalRateEstimator** — rolling-window λ per approach in veh/s (from `arrival_rate.py`)
3. **LSTM TurnPredictor** — predicts exit direction from partial trajectory (from `lstm_predictor.py`)

At every phase switch, these three signals are combined into a **demand score** per direction group (NS vs EW), and green time is split proportionally.

---

## Signal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_GREEN` | 10 s | Minimum green time for any phase |
| `MAX_GREEN` | 60 s | Maximum green time for any phase |
| `YELLOW_DUR` | 3 s | Yellow transition between phases |
| `CYCLE_TOTAL` | 90 s | Total green budget shared between NS and EW |

## Scoring Weights

| Weight | Value | Description |
|--------|-------|-------------|
| `W_QUEUE` | 1.0 | Weight for current queue count (vehicles stopped) |
| `W_ARRIVAL` | 3.0 | Weight for arrival rate signal |
| `W_THROUGH` | 0.3 | Bonus multiplier for through-traffic ratio |
| `ARRIVAL_LOOKAHEAD` | 10 s | Seconds to project arrival rate forward |

## QueueManager Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VELOCITY_THRESHOLD` | 1.0 m/s | Below this speed, vehicle is "queued" |
| `SMOOTHING_FRAMES` | 5 frames | Window for velocity averaging |
| `APPROACH_RADIUS` | 8 cells | Clustering radius for approach discovery |
| `MIN_APPROACH_COUNT` | 21 | Minimum trajectories to form an approach |
| `INTERSECTION_MARGIN` | 0.35 | Center 35% of grid excluded as intersection |

## ArrivalRateEstimator Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SECONDS` | 15 s | Rolling window for λ computation |

## LSTM Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 4 | Features per step: (x, y, dx, dy) |
| `hidden_size` | 64 | LSTM hidden units |
| `num_layers` | 2 | Stacked LSTM layers |
| `num_classes` | 4 | Exit zones (auto-discovered) |
| `dropout` | 0.3 | Dropout rate |
| `min_steps` | 3 | Minimum trajectory length before LSTM predicts |
| `confidence_threshold` | 0.4 | Below this, LSTM prediction is ignored |
| `cell_size` | 50 px | Grid cell size for trajectory discretization |

---

## Phase Cycle

```
NS_GREEN → NS_YELLOW (3s) → EW_GREEN → EW_YELLOW (3s) → repeat
```

The order is fixed. Only the **duration** of each green phase changes cycle to cycle.

---

## Decision Algorithm

At every phase transition (yellow → green), the controller:

### Step 1: Measure current queues
```
ns_queue = queued vehicles on N_to_C + S_to_C approaches
ew_queue = queued vehicles on E_to_C + W_to_C approaches
```
Uses QueueManager's velocity-based detection (speed < 1.0 m/s = stopped).

### Step 2: Measure arrival rates
```
ns_arrival = λ(North approach) + λ(South approach)   [veh/s]
ew_arrival = λ(East approach) + λ(West approach)     [veh/s]
```
Rolling 15-second window from ArrivalRateEstimator.

### Step 3: Predict through-traffic ratio via LSTM
For each active vehicle with ≥ 3 trajectory steps:
- Feed partial cell sequence to LSTM
- Get predicted exit zone + confidence
- If confidence ≥ 0.4, classify as through or turning
```
ns_through_ratio = (# NS vehicles predicted through) / (# NS vehicles predicted)
ew_through_ratio = (# EW vehicles predicted through) / (# EW vehicles predicted)
```
Default: 0.5 if no predictions available.

### Step 4: Compute demand scores
```
demand_score = (W_QUEUE × queue + W_ARRIVAL × arrival_rate × LOOKAHEAD) × (1 + W_THROUGH × through_ratio)
```

Expanded:
```
ns_score = (1.0 × ns_queue + 3.0 × ns_arrival × 10.0) × (1.0 + 0.3 × ns_through_ratio)
ew_score = (1.0 × ew_queue + 3.0 × ew_arrival × 10.0) × (1.0 + 0.3 × ew_through_ratio)
```

### Step 5: Split green time
```
ns_green = 90 × (ns_score / (ns_score + ew_score))
ew_green = 90 × (ew_score / (ns_score + ew_score))
```
Both clamped to [10, 60] seconds.

If both scores are 0: equal split (45s / 45s).

---

## Why each component matters

| Component | What it captures | Without it |
|-----------|-----------------|------------|
| **Queue count** | Immediate demand — vehicles stopped right now | Ignores current congestion |
| **Arrival rate** | Demand trend — is traffic building or subsiding? | Reacts too late to surges |
| **LSTM through-ratio** | Movement efficiency — through traffic clears faster than turns | Overallocates green to turn-heavy approaches |

---

## Experimental Results (seed=42)

| Scenario | Type | Avg Time (Fixed) | Avg Time (Adaptive) | Improvement |
|----------|------|------------------:|--------------------:|------------:|
| v1 | Balanced baseline | 56.65 s | 56.58 s | +0.1% |
| v2 | Demand flip | 60.48 s | 56.54 s | +6.5% |
| v3 | Morning CBD rush (NYC) | 64.51 s | 56.51 s | +12.4% |
| v4 | Evening exodus (LA) | 60.04 s | 55.87 s | +6.9% |
| v5 | School zone pulse | 44.59 s | 50.75 s | -13.8% |
| v6 | Oversaturated (Mumbai) | 63.12 s | 66.35 s | -5.1% |
| v7 | Stadium dispersal | 45.29 s | 37.71 s | +16.7% |

Adaptive wins 4/7 scenarios. Best on asymmetric/time-varying demand (v3, v7).
Fixed wins on short spikes (v5) and fully saturated conditions (v6).
