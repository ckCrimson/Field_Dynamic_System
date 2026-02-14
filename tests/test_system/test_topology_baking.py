import time
import pytest
import numpy as np
from typing import Any, Sequence

# Adjust imports to match your project structure
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.systems.static.topology import DiscreteStaticTopologySystem


# --- 1. Define the Custom Topology ---
class HexGridTopology(DiscreteTopology):
    def __init__(self):
        super().__init__(state_space=None)  # Pure Raw Mode

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        if hasattr(state_val, 'tolist'):
            state_val = tuple(state_val.tolist())
        elif isinstance(state_val, list):
            state_val = tuple(state_val)

        x, y = state_val
        return [
            (x + 1, y), (x - 1, y),
            (x, y + 1), (x - 1, y + 1),
            (x + 1, y - 1), (x, y - 1)
        ]


def test_topology_baking_and_iteration():
    print("\n=== TEST: Topology Baking & Iteration Benchmark ===")

    STEPS = 50
    # Expected volume = 3 * R * (R + 1) + 1
    EXPECTED_STATES = 3 * STEPS * (STEPS + 1) + 1
    print(f"Scenario: Expanding Hex Grid to {STEPS} steps (Expected States: {EXPECTED_STATES:,})")

    # Dummy space to satisfy the constructor references
    dummy_space = VectorStateSpace(vectors=[],dim=2)

    # -------------------------------------------------------
    # PATH 1: Object-Oriented Path Setup
    # -------------------------------------------------------
    print("\n[Path 1] Object-Oriented Setup")
    t0 = time.perf_counter()

    initial_obj = VectorState((0, 0))
    sys_oop = DiscreteStaticTopologySystem(
        initial_state=initial_obj,
        topology=HexGridTopology(),
        state_space=dummy_space
    )

    sys_oop.create_multi_step_states_space(steps=STEPS)
    t_oop_bake = time.perf_counter() - t0
    print(f"  - Bake Time (OOP Setup): {t_oop_bake:.4f}s")

    # -------------------------------------------------------
    # PATH 2: Raw Path Setup
    # -------------------------------------------------------
    print("\n[Path 2] Raw Path Setup")
    t0 = time.perf_counter()

    sys_raw = DiscreteStaticTopologySystem.from_raw_data(
        raw_initial_state=(0, 0),
        topology=HexGridTopology(),
        state_space=dummy_space,
        state_class=VectorState
    )

    sys_raw.create_multi_step_states_space(steps=STEPS)
    t_raw_bake = time.perf_counter() - t0
    print(f"  - Bake Time (Raw Setup): {t_raw_bake:.4f}s")

    # -------------------------------------------------------
    # ITERATION BENCHMARKS
    # -------------------------------------------------------
    print("\n--- Iteration Benchmarks ---")

    # We will do 10 passes over the data to magnify the timing differences
    PASSES = 100

    # 1. Traditional OOP Space Iteration
    # First, we must pay the heavy cost of instantiating all the objects
    t0 = time.perf_counter()
    space_oop = sys_raw.get_state_space()
    t_instantiate = time.perf_counter() - t0
    print(f"\n  [Method A] get_state_space() (Traditional OOP)")
    print(f"    - Cost to Instantiate Objects: {t_instantiate:.4f}s")

    t0 = time.perf_counter()
    for _ in range(PASSES):
        for state in space_oop.states:
            val = state.values  # Access the data
    t_iter_oop = time.perf_counter() - t0
    print(f"    - Iteration Time ({PASSES} passes): {t_iter_oop:.4f}s")

    # 2. Lazy Space Iteration
    t0 = time.perf_counter()
    space_lazy = sys_raw.get_lazy_state_space()
    t_lazy_setup = time.perf_counter() - t0
    print(f"\n  [Method B] get_lazy_state_space() (Hybrid/Proxy)")
    print(f"    - Cost to Instantiate Space: {t_lazy_setup:.6f}s (Near Zero!)")

    t0 = time.perf_counter()
    for _ in range(PASSES):
        for i in range(space_lazy.num_states):
            state = space_lazy.get_state_by_id(i)
            val = state.values  # Inflates object on the fly
    t_iter_lazy = time.perf_counter() - t0
    print(f"    - Iteration Time ({PASSES} passes): {t_iter_lazy:.4f}s")

    # 3. Pure Raw Iteration
    t0 = time.perf_counter()
    raw_array = sys_raw.get_raw_state_space()
    t_raw_setup = time.perf_counter() - t0
    print(f"\n  [Method C] get_raw_state_space() (Pure Data)")
    print(f"    - Cost to Instantiate Space: {t_raw_setup:.6f}s (Zero Copy!)")

    t0 = time.perf_counter()
    for _ in range(PASSES):
        # We can just iterate the numpy array directly
        for val in raw_array:
            pass  # Data is already raw
    t_iter_raw = time.perf_counter() - t0
    print(f"    - Iteration Time ({PASSES} passes): {t_iter_raw:.4f}s")

    # -------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------
    assert len(raw_array) == EXPECTED_STATES
    assert space_oop.num_states == EXPECTED_STATES
    assert space_lazy.num_states == EXPECTED_STATES

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    test_topology_baking_and_iteration()