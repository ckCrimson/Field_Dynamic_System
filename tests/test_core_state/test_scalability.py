import time
import sys
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Callable, Optional, Union, Set
from abc import ABC

import pytest

from src.field_dynamic_system.core import AbstractState
from src.field_dynamic_system.core.state import AbstractDiscreteStateSpace


# --- 3. THE BENCHMARK ---

def test_run_benchmark():
    print("=== BENCHMARK: AbstractState Scalability ===")

    # A. CONFIGURATION
    N = 100_000
    print(f"Generating {N:,} raw string inputs...")
    raw_strings = [f"state_{i}" for i in range(N)]

    # --- SCENARIO 1: The Standard OOP Way (Slow) ---
    print("\n[Method A] Standard Init (Object Creation + Sorting)")

    t_start_oop = time.time()

    # Step 1: Create Objects
    objects = [AbstractState(name=s) for s in raw_strings]

    # Step 2: Initialize Space
    try:
        space_oop = AbstractDiscreteStateSpace(states=objects)
    except Exception as e:
        pytest.fail(f"OOP Init failed: {e}")

    t_end_oop = time.time()
    total_time_oop = t_end_oop - t_start_oop

    print(f"  - Total Time: {total_time_oop:.4f}s")

    # --- SCENARIO 2: The Factory Way (Fast) ---
    print("\n[Method B] Factory Init (Raw Data Injection)")

    t_start_factory = time.time()

    # Step 1: Direct Injection
    space_factory = AbstractDiscreteStateSpace.from_raw_data(
        raw_data=raw_strings,
        wrapper=AbstractState
    )

    t_end_factory = time.time()
    total_time_factory = t_end_factory - t_start_factory

    print(f"  - Total Time:      {total_time_factory:.6f}s")

    # --- ASSERTIONS ---

    # 1. Speed Check
    # Avoid division by zero if factory is instant
    speedup = total_time_oop / (total_time_factory + 1e-9)
    print(f"  - Speedup Factor:  {speedup:.1f}x")

    # Benchmark Assertion: Factory must be faster
    assert total_time_factory < total_time_oop, "Factory init should be faster than OOP init"

    # 2. Correctness Check
    print("\n[Verification]")

    # Verify Factory created a valid space
    assert space_factory.num_states == N

    # Verify Lazy Access works
    first_state = space_factory.get_state_by_id(0)
    assert isinstance(first_state, AbstractState)
    assert first_state.name == "state_0"

    print("  - State 0 name verified: 'state_0'")


if __name__ == "__main__":
    test_run_benchmark()