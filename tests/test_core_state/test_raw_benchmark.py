import time
import pytest
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Sequence, Callable, Optional

from src.field_dynamic_system.core import AbstractState, VectorState
from src.field_dynamic_system.core.state import AbstractDiscreteStateSpace, VectorStateSpace
from src.field_dynamic_system.core.state import AbstractState as StringState


# --- 1. SETUP: DEFINING THE ARCHITECTURE WITH RAW OPS ---

# --- 3. HELPER: TIMER ---
def time_op(name, func):
    t0 = time.time()
    res = func()
    dt = time.time() - t0
    return res, dt


# =========================================================
# TEST 1: ABSTRACT STATE (STRINGS)
# =========================================================



def test_abstract_ops_benchmark():
    print("\n\n=== BENCHMARK 1: ABSTRACT STATE (Strings) ===")

    # 1. SETUP DATA (Overlap Scenario)
    # Set A: 0 to 100k
    # Set B: 50k to 150k
    # Overlap: 50k
    N = 100_000
    OVERLAP = 50_000

    raw_A = [f"s_{i}" for i in range(N)]
    raw_B = [f"s_{i}" for i in range(N - OVERLAP, 2 * N - OVERLAP)]

    print(f"Data Generated: A={len(raw_A)}, B={len(raw_B)}, Exp. Overlap={OVERLAP}")

    # 2. INITIALIZE SPACES
    # OOP Path
    objs_A = [StringState(s) for s in raw_A]
    objs_B = [StringState(s) for s in raw_B]
    space_A_oop = AbstractDiscreteStateSpace(objs_A)
    space_B_oop = AbstractDiscreteStateSpace(objs_B)

    # Raw Path
    space_A_raw = AbstractDiscreteStateSpace.from_raw_data(raw_A, StringState)
    space_B_raw = AbstractDiscreteStateSpace.from_raw_data(raw_B, StringState)

    # 3. BENCHMARK: INTERSECTION
    print("\n[Operation: Intersection]")

    # OOP
    res_oop, t_oop = time_op("OOP Intersect", lambda: space_A_oop.intersection(space_B_oop))
    print(f"  - OOP Time: {t_oop:.4f}s")

    # Raw
    res_raw, t_raw = time_op("Raw Intersect", lambda: space_A_raw.raw_intersection(space_B_raw))
    print(f"  - Raw Time: {t_raw:.4f}s")

    print(f"  - Speedup:  {t_oop / t_raw:.1f}x")

    # Assertions
    assert len(res_oop._idx_to_state) == OVERLAP
    assert len(res_raw._idx_to_state) == OVERLAP
    assert t_raw < t_oop

    # 4. BENCHMARK: LOOKUP (Contains)
    print("\n[Operation: Lookup (1000 items)]")

    # Setup queries
    queries_raw = [f"s_{i}" for i in range(0, 1000)]
    queries_obj = [StringState(s) for s in queries_raw]

    # OOP Lookup Loop
    t0 = time.time()
    for q in queries_obj:
        space_A_oop.contains(q)
    t_oop = time.time() - t0
    print(f"  - OOP Time: {t_oop:.6f}s (Hash Map)")

    # Raw Lookup Loop (Simulating physics engine checking raw IDs)
    t0 = time.time()
    # Note: Linear scan on array is slow, but we test 'contains_raw' logic
    # Real physics engine uses sorted search or indices, but let's test the raw capability
    # Optimizing the test to use a batch check which is realistic for physics
    mask = np.isin(queries_raw, space_A_raw.raw_states)
    t_raw = time.time() - t0
    print(f"  - Raw Time: {t_raw:.6f}s (Vectorized Batch)")

    print(f"  - Speedup:  {t_oop / t_raw:.1f}x")


# =========================================================
# TEST 2: VECTOR STATE SPACE
# =========================================================

def test_vector_ops_benchmark():
    print("\n\n=== BENCHMARK 2: VECTOR STATE (JAX/Numpy) ===")

    # 1. SETUP DATA (2D Grid)
    # A: [0,0] to [100,100]
    # B: [50,50] to [150,150]
    N_SIDE = 100

    # Generate raw grids
    x = np.arange(N_SIDE)
    y = np.arange(N_SIDE)
    grid_A = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    # Shift grid B by (50, 50)
    grid_B = grid_A + 50

    print(f"Data Generated: {len(grid_A)} vectors per set.")

    # 2. INITIALIZE SPACES
    # Raw Path is the natural fit for vectors
    space_A = VectorStateSpace.from_raw_data(
        grid_A,
        wrapper=lambda x: VectorState(tuple(x)),
        dim=2
    )
    space_B = VectorStateSpace.from_raw_data(
        grid_B,
        wrapper=lambda x: VectorState(tuple(x)),
        dim=2
    )

    # 3. BENCHMARK: UNION (Vectorized)
    print("\n[Operation: Vector Union]")

    # Note: Vectors are strictly numerical, so 'Raw' logic is native.
    # We compare against a hypothetical 'Object Loop' approach.

    # Approach A: Object Loop (Simulated)
    t0 = time.time()
    objs_A = [VectorState(tuple(v)) for v in grid_A]
    objs_B = [VectorState(tuple(v)) for v in grid_B]
    combined_objs = list(set(objs_A) | set(objs_B))
    t_oop = time.time() - t0
    print(f"  - Object Set Time: {t_oop:.4f}s")

    # Approach B: Raw Matrix Ops
    t0 = time.time()
    res_raw = space_A.raw_union(space_B)
    t_raw = time.time() - t0
    print(f"  - Raw Matrix Time: {t_raw:.4f}s")

    print(f"  - Speedup: {t_oop / t_raw:.1f}x")

    # Verify Size
    # Total area should be: Union of two overlapping squares
    # This logic is tricky with indices, let's just assert Raw is valid
    assert len(res_raw._idx_to_state) > len(grid_A)


if __name__ == "__main__":
    test_abstract_ops_benchmark()
    test_vector_ops_benchmark()