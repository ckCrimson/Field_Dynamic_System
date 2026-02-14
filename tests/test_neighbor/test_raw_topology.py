import time
import pytest
import numpy as np
from typing import Any, Sequence

# Adjust imports to match your structure
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.systems.static.topology import DiscreteStaticTopologySystem


# 1. Define the Custom Topology
class HexGridTopology(DiscreteTopology):
    """
    A 2D topology with 6-fold symmetry (Hexagonal Grid on axial coordinates).
    Neighbors: +x, -x, +y, -y, (+x, -y), (-x, +y)
    """

    def __init__(self):
        # Pass None for state_space to enforce Pure Raw Mode
        super().__init__(state_space=None)

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        # Ensure the state is a hashable tuple (in case numpy arrays are passed)
        if hasattr(state_val, 'tolist'):
            state_val = tuple(state_val.tolist())
        elif isinstance(state_val, list):
            state_val = tuple(state_val)

        x, y = state_val
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1)
        ]


def test_raw_topology_path():
    print("\n=== TEST: Pure Raw Topology (Zero Object) ===")

    # 1. Setup the System
    # Initial state is origin (0,0)
    initial_raw = (0, 0)
    topology = HexGridTopology()

    # Initialize using the Fast Factory (No StateSpace provided!)
    sys = DiscreteStaticTopologySystem.from_raw_data(
        raw_initial_state=initial_raw,
        topology=topology
    )

    # -------------------------------------------------------
    # TEST 1: The 1-Step Successor (Validation)
    # -------------------------------------------------------
    print("\n[Test 1] Single Step Expansion")

    # Run 1 step
    step_1_results = sys.get_raw_multistep(steps=1)

    print(f"  - Initial State: {initial_raw}")
    print(f"  - Found Neighbors ({len(step_1_results)}): {step_1_results}")

    # Verify we found exactly 6 unique neighbors
    assert len(step_1_results) == 6, f"Expected 6 neighbors, got {len(step_1_results)}"

    expected_neighbors = set([(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)])
    actual_neighbors = set([tuple(s) if hasattr(s, 'tolist') else s for s in step_1_results])
    assert actual_neighbors == expected_neighbors, "Neighbor coordinates do not match expected hex grid."

    # Verify the CPU matrix dynamically resized itself!
    # Origin + 6 neighbors = 7 total explored states mapped in the matrix so far.
    matrix_shape = topology._raw_cpu_matrix.shape
    print(f"  - Dynamic Matrix Shape: {matrix_shape}")
    assert matrix_shape[0] >= 7

    # -------------------------------------------------------
    # TEST 2: The Multi-Step Benchmark (20 Steps)
    # -------------------------------------------------------
    print("\n[Test 2] Multi-Step Benchmark (20 Steps)")

    steps = 89
    t0 = time.time()

    # Ask for reachable states within 20 steps
    # Because edges are bidirectional, the successors of the frontier
    # will bleed backward, eventually returning the entire solid volume.
    step_20_volume = sys.get_raw_multistep(steps=steps)

    t_raw = time.time() - t0

    # Hex grid total area/volume for radius R is 3*R*(R+1) + 1
    expected_volume_size = 3 * steps * (steps + 1) + 1

    print(f"  - Time: {t_raw:.4f}s")
    print(f"  - Reachable Volume at Step {steps}: {len(step_20_volume)}")

    assert len(step_20_volume) == expected_volume_size, \
        f"Expected {expected_volume_size} states, got {len(step_20_volume)}"

    # Check the total volume discovered by the Topology cache
    total_discovered = len(topology._id_to_raw)
    print(f"  - Total States Discovered & Cached: {total_discovered} (Expected: {expected_volume_size})")

    assert total_discovered == expected_volume_size


if __name__ == "__main__":
    test_raw_topology_path()