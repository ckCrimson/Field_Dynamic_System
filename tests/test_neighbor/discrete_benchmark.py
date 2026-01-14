import time
import jax
from typing import List

from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# --- 1. SETUP: The Custom Topology ---
class ThickLineTopology(DiscreteTopology):
    """
    Connects s to [s-2, s+2].
    This creates a 'band' matrix (width 5).
    """

    def __init__(self, space, N):
        super().__init__(space)
        self.N = N

    def compute_neighbors(self, state: int) -> List[int]:
        # User Logic: simple range arithmetic
        # We clamp values to stay within [0, N-1]
        start = max(0, state - 2)
        end = min(self.N - 1, state + 2)

        # Return all integers in range [start, end] inclusive
        return list(range(start, end + 1))


# --- 2. BENCHMARK ---
def test_benchmark_dense_line_propagation(capsys):
    N_STATES = 10_000
    STEPS = 50

    print(f"\n\n--- Benchmark: Thick Line (N={N_STATES}) ---")

    # 1. Create Space
    print("Generating State Space...")
    states = list(range(N_STATES))
    space = AbstractDiscreteStateSpace(states)

    # 2. Create Topology
    topology = ThickLineTopology(space, N=N_STATES)

    # 3. Measure Compilation Time (First Run)
    # This includes:
    #   a) Calling user logic 10,000 times (Python loop)
    #   b) Building the JAX Matrix
    #   c) Compiling the JAX Multi-step Kernel
    print("Building Matrix & Compiling JAX...")

    start_time = time.time()
    # We trigger compilation by asking for the adjacency matrix explicitly
    # or running a dummy step. Let's run a dummy step.
    _ = topology.multi_step_successor(0, steps=1).states
    build_duration = time.time() - start_time

    print(f"Compilation/Build Time: {build_duration:.4f}s")

    # 4. Measure Execution Time (Hot Run)
    # We want to calculate 50 steps.
    # Mathematically: A single step expands bounds by 2.
    # 50 steps should expand bounds by 100.
    # Range [0, 0] -> [0, 100].

    print(f"Running {STEPS} Steps Propagation...")

    start_node = N_STATES // 2  # Start in the middle (5000)

    # Warmup JIT for the specific 'steps=50' path if strictly needed,
    # but usually 'fori_loop' is compiled once.
    # We block until ready to ensure GPU/CPU sync.
    jax.block_until_ready(topology.adjacency_matrix)

    start_t = time.time()
    result_space = topology.multi_step_successor(start_node, steps=STEPS)

    # Force computation
    result_count = result_space.num_states
    duration = time.time() - start_t

    print(f"Execution Time ({STEPS} steps): {duration * 1000:.4f} ms")
    print(f"Reached States: {result_count}")

    # --- Verification ---
    # Radius grows by 2 every step.
    # Step 50 -> Radius 100.
    # Range: [5000-100, 5000+100] -> width 201 (inclusive)
    expected_count = 201
    assert result_count == expected_count, f"Expected {expected_count}, got {result_count}"