import time
import pytest
import numpy as np

# Adjust imports to your structure
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.systems.static.state import  DiscreteStaticStateSystem


def test_static_system_benchmark():
    print("\n=== TEST: Static System Initialization Benchmark ===")

    # 1. SETUP SCENARIO
    N_AGENTS = 1_000_000  # 1 Million Agents
    DIM = 3  # X, Y, Z
    print(f"Scenario: {N_AGENTS:,} Agents in 3D Space")

    # Create Space (Fast Path)
    # 200x200x200 grid is HUGE (8M states), so we use a smaller logic space
    # but initialize agents at (0,0,0) which exists.
    # We cheat and use a small raw buffer for the space just to make it valid.
    space_raw = np.array([[0, 0, 0], [1, 1, 1]])
    space = VectorStateSpace.from_raw_data(space_raw, lambda x: VectorState(tuple(x)), DIM)

    # 2. DEFINE INPUTS

    # Input A: Object (Single VectorState)
    # "All agents start at Origin"
    initial_obj = VectorState((0, 0, 0))

    # Input B: Raw Data (Single Tuple)
    initial_raw = np.array([0, 0, 0])

    # -------------------------------------------------------
    # PATH A: The "Slow" Object Way (Simulation)
    # -------------------------------------------------------
    print("\n[Path A] Standard Init (Object Broadcast)")

    # To simulate the "Bad" way, we manually create a list of 1 million objects
    # because the standard constructor usually expects a list if we want N entities.
    # If we pass a single object, it just stores 1 object.
    # So we must simulate the user creating a list of objects.

    t0 = time.time()

    # 1. Create List of Objects (Simulating user input)
    # This is the bottleneck we are trying to avoid!
    object_list = [initial_obj] * N_AGENTS

    # 2. Initialize System
    sys_obj = DiscreteStaticStateSystem(object_list, space)

    t_obj = time.time() - t0
    print(f"  - Time: {t_obj:.4f}s")

    # -------------------------------------------------------
    # PATH B: The "Fast" Raw Way (Factory Broadcast)
    # -------------------------------------------------------
    print("\n[Path B] Raw Factory (Numpy Broadcast)")
    t0 = time.time()

    # 1. Factory Call
    # We pass the single raw vector and tell it "Make 1 Million of these"
    # It uses np.tile() internally.
    sys_raw = DiscreteStaticStateSystem.from_raw_data(
        raw_initial_state=initial_raw,
        state_space=space,
        num_entities=N_AGENTS
    )

    t_raw = time.time() - t0
    print(f"  - Time: {t_raw:.4f}s")

    # -------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------
    print("\n[Verification]")

    # 1. Speedup
    speedup = t_obj / (t_raw + 1e-9)
    print(f"  - Speedup: {speedup:.1f}x")

    # Creating 1M objects vs Tiling 1 numpy array -> Expect 50x+ speedup
    assert t_raw < t_obj

    # 2. Data Integrity
    # Check shape
    raw_config = sys_raw.get_raw_state()
    print(f"  - Raw Config Shape: {raw_config.shape}")

    assert raw_config.shape == (N_AGENTS, DIM)

    # Check content of agent #999,999
    last_agent = raw_config[-1]
    print(f"  - Last Agent Pos: {last_agent}")
    assert np.array_equal(last_agent, [0, 0, 0])


if __name__ == "__main__":
    test_static_system_benchmark()