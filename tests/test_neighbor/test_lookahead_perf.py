import time
import numpy as np
import pytest
import scipy.sparse as sp
from typing import Sequence, Any, List

from src.field_dynamic_system.core.state import AbstractState, AbstractDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# =========================================================
# 1. CONCRETE TOPOLOGY
# =========================================================

class NumberLineTopology(DiscreteTopology):
    """
    Simple 1D Line Topology: x connects to [x-1, x+1]
    Expanding 10 steps deep creates a line of -10 to +10.
    """

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        val = int(state_val)
        return [str(val - 1), str(val + 1)]


# =========================================================
# 2. THE PERFORMANCE TEST
# =========================================================

def test_lookahead_vs_iterative():
    print("\n=========================================================")
    print("🚀 BENCHMARK: Matrix Build (Iterative vs Lookahead)")
    print("=========================================================")

    # --- SETUP 1: Iterative Topology ---
    space1 = AbstractDiscreteStateSpace([AbstractState("0", {})])
    topo_iterative = NumberLineTopology(space1)

    # --- SETUP 2: Lookahead Topology ---
    space2 = AbstractDiscreteStateSpace([AbstractState("0", {})])
    topo_lookahead = NumberLineTopology(space2)

    DEPTH = 50  # Deep expansion to stress the matrix resize logic
    start_node = ["0"]

    # ---------------------------------------------------------
    # METHOD A: ITERATIVE (Step-by-Step Physics)
    # ---------------------------------------------------------
    # This simulates running physics step 1, expanding, running step 2, expanding...
    print(f"\n[A] Running Iterative Expansion (Depth {DEPTH})...")

    t0 = time.time()
    current_frontier = start_node

    for _ in range(DEPTH):
        # This triggers neighbor compute + matrix update + (implicitly) JAX sync if accessed
        next_nodes = topo_iterative.get_raw_successor(current_frontier)
        current_frontier = next_nodes

        # Simulate accessing the matrix (forcing sync cost)
        _ = topo_iterative._raw_jax_matrix

    t_iterative = time.time() - t0
    print(f"⏱️  Time: {t_iterative:.4f}s")
    print(f"   Final Matrix Size: {topo_iterative._raw_cpu_matrix.shape}")

    # ---------------------------------------------------------
    # METHOD B: LOOKAHEAD (Scout & Solve)
    # ---------------------------------------------------------
    # This simulates scouting 50 steps ahead, building matrix ONCE, then syncing ONCE.
    print(f"\n[B] Running Lookahead Expansion (Depth {DEPTH})...")

    t0 = time.time()

    # 1. SCOUT: Pure CPU BFS
    # Note: You need to add the 'expand_frontier' method I provided earlier
    # to your DiscreteTopology class for this to work.
    topo_lookahead.expand_frontier(start_node, depth=DEPTH)

    # 2. SYNC: Upload once
    topo_lookahead._raw_sync_to_device()

    t_lookahead = time.time() - t0
    print(f"⏱️  Time: {t_lookahead:.4f}s")
    print(f"   Final Matrix Size: {topo_lookahead._raw_cpu_matrix.shape}")

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print(f"\n--- Results ---")
    if t_iterative > 0:
        speedup = t_iterative / t_lookahead
        print(f"🚀 Speedup: {speedup:.1f}x")

    # Sanity Check: Matrices should be roughly same size (covering same area)
    size_a = topo_iterative._raw_cpu_matrix.nnz
    size_b = topo_lookahead._raw_cpu_matrix.nnz
    print(f"   Edges Found (A): {size_a}")
    print(f"   Edges Found (B): {size_b}")

    assert size_b >= size_a, "Lookahead failed to discover the full graph!"


if __name__ == "__main__":
    test_lookahead_vs_iterative()