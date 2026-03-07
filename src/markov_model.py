"""
Markov Chain Intention Predictor
=================================
Usage:
    python markov_model.py --data control_transitions.json

Takes the transition data you collected and:
    1. Builds a Markov chain transition matrix
    2. Auto-discovers exit zones by clustering endpoints
    3. Computes P(exit | cell) for every cell using simulation
    4. Prints a full report
    5. Saves the trained model for use in real-time prediction

No manual exit zone definition needed — everything is learned from data.
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load transition data
# ═══════════════════════════════════════════════════════════════════════════

def load_transitions(filepath):
    """Load the transition JSON from the tracker."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    cell_size = data["cell_size"]
    cells = {}
    for key, val in data["cells"].items():
        cx, cy = map(int, key.split(","))
        neighbors = {}
        for n_key, n_val in val["neighbors"].items():
            nx, ny = map(int, n_key.split(","))
            neighbors[(nx, ny)] = {
                "count": n_val["count"],
                "probability": n_val["probability"]
            }
        cells[(cx, cy)] = {
            "total": val["total_transitions"],
            "neighbors": neighbors
        }

    endpoints = []
    for ep in data["endpoints"]:
        endpoints.append({
            "cell": tuple(ep["cell"]),
            "track_id": ep["track_id"],
            "frame": ep["frame"]
        })

    startpoints = []
    for sp in data["startpoints"]:
        startpoints.append({
            "cell": tuple(sp["cell"]),
            "track_id": sp["track_id"],
            "frame": sp["frame"]
        })

    print(f"Loaded: {len(cells)} cells, {len(endpoints)} endpoints, {len(startpoints)} startpoints")
    return cell_size, cells, endpoints, startpoints


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Build Markov Transition Matrix
# ═══════════════════════════════════════════════════════════════════════════

class MarkovChain:
    """
    Markov chain built from cell transition data.

    States = grid cells
    Transition probabilities = P(next_cell | current_cell)
    Absorbing states = cells where vehicles disappear (endpoints)
    """

    def __init__(self, cells):
        self.cells = cells
        self.all_states = set(cells.keys())

        # Also add neighbor cells as states (they might not have their own transitions)
        for cell, data in cells.items():
            for neighbor in data["neighbors"]:
                self.all_states.add(neighbor)

        print(f"Markov chain: {len(self.all_states)} total states")

    def get_transition_probs(self, cell):
        """Get P(c'|c) for a cell. Returns dict of {neighbor: prob}."""
        if cell not in self.cells or self.cells[cell]["total"] == 0:
            return {}
        return {
            n: v["probability"]
            for n, v in self.cells[cell]["neighbors"].items()
            if v["probability"] > 0
        }

    def simulate_path(self, start_cell, max_steps=100):
        """
        Simulate a vehicle path from start_cell using the Markov chain.
        Follows probabilities until reaching a dead-end (absorbing state).

        Returns:
            list of cells visited
        """
        path = [start_cell]
        current = start_cell
        visited = set()

        for _ in range(max_steps):
            probs = self.get_transition_probs(current)
            if not probs:
                break  # Dead end — absorbing state

            # Avoid infinite loops
            if current in visited and len(visited) > 3:
                break
            visited.add(current)

            # Sample next cell based on probabilities
            neighbors = list(probs.keys())
            weights = list(probs.values())

            # Normalize weights (should already sum to ~1)
            total_w = sum(weights)
            if total_w == 0:
                break
            weights = [w / total_w for w in weights]

            next_cell = neighbors[np.random.choice(len(neighbors), p=weights)]
            path.append(next_cell)
            current = next_cell

        return path

    def predict_destination(self, start_cell, n_simulations=500, max_steps=100):
        """
        Run Monte Carlo simulations to predict where a vehicle starting
        at start_cell will end up.

        Returns:
            dict: {endpoint_cell: probability}
        """
        endpoint_counts = defaultdict(int)

        for _ in range(n_simulations):
            path = self.simulate_path(start_cell, max_steps)
            endpoint = path[-1]  # Where the vehicle ended up
            endpoint_counts[endpoint] += 1

        # Normalize to probabilities
        total = sum(endpoint_counts.values())
        return {cell: count / total for cell, count in endpoint_counts.items()}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Auto-discover exit zones from endpoints
# ═══════════════════════════════════════════════════════════════════════════

def cluster_endpoints(endpoints, radius=2):
    """
    Simple clustering of endpoint cells.
    Groups endpoints that are within `radius` cells of each other.

    No sklearn needed — just a basic distance-based merge.

    Returns:
        list of clusters: [{"center": (cx,cy), "cells": set(), "count": int, "label": str}, ...]
    """
    # Count endpoints per cell
    ep_counts = defaultdict(int)
    for ep in endpoints:
        ep_counts[ep["cell"]] += 1

    if not ep_counts:
        print("No endpoints to cluster!")
        return []

    # Sort by count (most popular first)
    sorted_cells = sorted(ep_counts.items(), key=lambda x: x[1], reverse=True)

    clusters = []
    assigned = set()

    for cell, count in sorted_cells:
        if cell in assigned:
            continue

        # Start a new cluster
        cluster_cells = set()
        cluster_count = 0

        # Find all nearby cells
        for other_cell, other_count in sorted_cells:
            if other_cell in assigned:
                continue
            dist = abs(cell[0] - other_cell[0]) + abs(cell[1] - other_cell[1])
            if dist <= radius:
                cluster_cells.add(other_cell)
                cluster_count += other_count
                assigned.add(other_cell)

        if cluster_count >= 2:  # Minimum 2 vehicles to form a cluster
            # Compute center
            cx = sum(c[0] for c in cluster_cells) / len(cluster_cells)
            cy = sum(c[1] for c in cluster_cells) / len(cluster_cells)
            clusters.append({
                "center": (round(cx, 1), round(cy, 1)),
                "cells": cluster_cells,
                "count": cluster_count,
            })

    # Sort by count and assign labels
    clusters.sort(key=lambda x: x["count"], reverse=True)
    for i, cl in enumerate(clusters):
        cl["label"] = f"exit_{i}"

    return clusters


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Compute P(exit_zone | cell) for every cell
# ═══════════════════════════════════════════════════════════════════════════

def compute_exit_probabilities(markov, cells, exit_clusters, n_simulations=300):
    """
    For each cell with data, run simulations to compute
    P(exit_zone | cell) — probability of reaching each exit zone.

    Returns:
        dict: {cell: {exit_label: probability, ...}}
    """
    # Build a lookup: cell → exit label
    cell_to_exit = {}
    for cluster in exit_clusters:
        for c in cluster["cells"]:
            cell_to_exit[c] = cluster["label"]

    exit_labels = [cl["label"] for cl in exit_clusters]
    result = {}

    # Only compute for cells that have meaningful traffic
    active_cells = [c for c, d in cells.items() if d["total"] >= 2]
    print(f"\nComputing P(exit | cell) for {len(active_cells)} cells "
          f"with {n_simulations} simulations each...")

    for i, cell in enumerate(active_cells):
        destinations = markov.predict_destination(cell, n_simulations)

        # Map destinations to exit zones
        exit_probs = {label: 0.0 for label in exit_labels}
        unmapped = 0.0

        for dest_cell, prob in destinations.items():
            if dest_cell in cell_to_exit:
                exit_probs[cell_to_exit[dest_cell]] += prob
            else:
                # Check if near any exit cluster
                matched = False
                for cluster in exit_clusters:
                    for ec in cluster["cells"]:
                        if abs(dest_cell[0] - ec[0]) <= 1 and abs(dest_cell[1] - ec[1]) <= 1:
                            exit_probs[cluster["label"]] += prob
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    unmapped += prob

        exit_probs["unknown"] = unmapped
        result[cell] = exit_probs

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(active_cells)} cells...")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Report & Analysis
# ═══════════════════════════════════════════════════════════════════════════

def print_report(cells, exit_clusters, exit_probs, startpoints):
    """Print a comprehensive analysis report."""

    print(f"\n{'='*70}")
    print(f"  MARKOV CHAIN INTENTION MODEL — FULL REPORT")
    print(f"{'='*70}")

    # Exit zone summary
    print(f"\n── Auto-Discovered Exit Zones ──")
    for cl in exit_clusters:
        print(f"  {cl['label']}: center={cl['center']}, "
              f"vehicles={cl['count']}, cells={cl['cells']}")

    # Startpoint summary
    sp_counts = defaultdict(int)
    for sp in startpoints:
        sp_counts[sp["cell"]] += 1
    sorted_sp = sorted(sp_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n── Entry Points (where vehicles appear) ──")
    for cell, count in sorted_sp[:10]:
        print(f"  Cell {cell}: {count} vehicles entered")

    # P(exit | cell) for key cells
    print(f"\n── P(exit | cell) for High-Traffic Cells ──")
    sorted_cells = sorted(
        [(c, d) for c, d in cells.items() if d["total"] >= 3],
        key=lambda x: x[1]["total"],
        reverse=True
    )

    for cell, data in sorted_cells[:20]:
        if cell not in exit_probs:
            continue
        probs = exit_probs[cell]
        # Filter out near-zero
        sig_probs = {k: v for k, v in probs.items() if v > 0.02}
        prob_str = " | ".join([f"{k}: {v:.0%}" for k, v in
                               sorted(sig_probs.items(), key=lambda x: x[1], reverse=True)])
        dominant = max(sig_probs, key=sig_probs.get) if sig_probs else "?"
        print(f"  Cell {cell} (N={data['total']:>3}): [{prob_str}]  → likely: {dominant}")

    # Flow analysis
    print(f"\n── Traffic Flow Chains (most likely paths) ──")
    entry_cells = [cell for cell, count in sorted_sp[:5]]
    for start in entry_cells:
        if start not in cells:
            continue
        # Follow the most likely path
        path = [start]
        current = start
        for _ in range(15):
            probs = {n: v["probability"] for n, v in cells.get(current, {}).get("neighbors", {}).items()
                     if v["probability"] > 0}
            if not probs:
                break
            best = max(probs, key=probs.get)
            path.append(best)
            current = best

        path_str = " → ".join([f"({c[0]},{c[1]})" for c in path])
        print(f"  From {start}: {path_str}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Save trained model
# ═══════════════════════════════════════════════════════════════════════════

def save_model(filepath, cell_size, exit_clusters, exit_probs, cells):
    """Save the trained Markov model for real-time use."""
    model_data = {
        "cell_size": cell_size,
        "exit_zones": [],
        "cell_predictions": {},
        "flow_chains": {}
    }

    # Exit zones
    for cl in exit_clusters:
        model_data["exit_zones"].append({
            "label": cl["label"],
            "center": list(cl["center"]),
            "cells": [list(c) for c in cl["cells"]],
            "count": cl["count"]
        })

    # P(exit | cell) for each cell
    for cell, probs in exit_probs.items():
        key = f"{cell[0]},{cell[1]}"
        model_data["cell_predictions"][key] = {
            k: round(v, 4) for k, v in probs.items() if v > 0.01
        }

    # Most likely next cell for each cell (for real-time path prediction)
    for cell, data in cells.items():
        if data["total"] < 2:
            continue
        key = f"{cell[0]},{cell[1]}"
        best_neighbor = None
        best_prob = 0
        for n, v in data["neighbors"].items():
            if v["probability"] > best_prob:
                best_prob = v["probability"]
                best_neighbor = n
        if best_neighbor:
            model_data["flow_chains"][key] = {
                "next": list(best_neighbor),
                "probability": round(best_prob, 4)
            }

    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"\n  Trained model saved to: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Markov Chain Intention Predictor")
    parser.add_argument("--data", required=True, help="Path to transitions JSON")
    parser.add_argument("--simulations", type=int, default=300,
                        help="Monte Carlo simulations per cell (default: 300)")
    parser.add_argument("--cluster_radius", type=int, default=2,
                        help="Endpoint clustering radius in cells (default: 2)")
    args = parser.parse_args()

    # Step 1: Load data
    print("\n── Step 1: Loading transition data ──")
    cell_size, cells, endpoints, startpoints = load_transitions(args.data)

    # Step 2: Build Markov chain
    print("\n── Step 2: Building Markov chain ──")
    markov = MarkovChain(cells)

    # Step 3: Cluster endpoints to discover exit zones
    print("\n── Step 3: Auto-discovering exit zones ──")
    exit_clusters = cluster_endpoints(endpoints, radius=args.cluster_radius)
    print(f"Found {len(exit_clusters)} exit zones:")
    for cl in exit_clusters:
        print(f"  {cl['label']}: center={cl['center']}, "
              f"vehicles={cl['count']}, cells={sorted(cl['cells'])}")

    # Step 4: Compute P(exit | cell)
    print("\n── Step 4: Computing exit probabilities ──")
    exit_probs = compute_exit_probabilities(
        markov, cells, exit_clusters, n_simulations=args.simulations
    )

    # Step 5: Print report
    print_report(cells, exit_clusters, exit_probs, startpoints)

    # Step 6: Save model
    output_path = Path(args.data).stem + "_markov_model.json"
    save_model(output_path, cell_size, exit_clusters, exit_probs, cells)

    print(f"\n{'='*70}")
    print(f"  DONE! Model ready for real-time prediction.")
    print(f"  Load '{output_path}' in your tracking pipeline to predict")
    print(f"  vehicle intentions as they enter the intersection.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()