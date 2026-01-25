import time
import numpy as np
import pytest
from typing import Sequence, Any

# Adjust imports to match your project structure
from src.field_dynamic_system.core.state import  AbstractDiscreteStateSpace, AbstractState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# =========================================================
# 1. CONCRETE TOPOLOGY (Primitive Logic)
# =========================================================

class BenchmarkTopology(DiscreteTopology):
    """
    Implements neighbor logic using PRIMITIVES (Strings/Ints).
    This ensures the Raw Track is not slowed down by Object creation.
    """

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        # Input: "10" (str)
        # Output: ["8", "9", "10", "11", "12"] (List[str])

        # Fast conversion for math
        val = int(state_val)

        return [
            str(val - 2),
            str(val - 1),
            str(val),
            str(val + 1),
            str(val + 2)
        ]


# =========================================================
# 2. THE CORRECTED TEST
# =========================================================

def test_run_million_state_benchmark():
    print("\n=========================================================")
    print("🚀 BENCHMARK: 1 MILLION STATES (Object vs Raw Matrix)")
    print("=========================================================")

    # 1. SETUP
    # We initialize the space with a dummy state to satisfy __init__
    initial = [AbstractState("0", {})]
    space = AbstractDiscreteStateSpace(initial)
    topology = BenchmarkTopology(space)

    # 2. GENERATE INPUTS
    N_STATES = 100_000  # Keep at 100k for CI/Test speed. Scale to 1M for full stress test.
    print(f"-> Generating {N_STATES} random inputs...")

    random_ints = np.random.randint(0, 10_000_000, N_STATES)

    # CRITICAL: Create two datasets
    # Set A: Primitive Strings (For Raw Track) -> ["1", "50", ...]
    input_primitives = [str(i) for i in random_ints]

    # Set B: State Objects (For Legacy Baseline) -> [AbstractState("1"), ...]
    input_objects = [AbstractState(p, {}) for p in input_primitives]

    # ---------------------------------------------------------
    # A. OBJECT BASELINE (The "Old Way")
    # ---------------------------------------------------------
    print(f"\n[A] Running Object Logic (Python Loop + Object Wrapper)...")
    t0 = time.time()

    object_results = set()

    # Simulate the legacy behavior:
    # Iterate Objects -> Get Value -> Compute -> Wrap in Objects
    for s_obj in input_objects:
        # 1. Extract Primitive (Simulating what the Topology would do)
        val = s_obj.name

        # 2. Compute Neighbors (Primitive)
        neighbors_raw = topology.compute_neighbors(val)

        # 3. Wrap back into Objects (The cost of the Object System)
        for n_val in neighbors_raw:
            object_results.add(AbstractState(n_val, {}))

    t_object = time.time() - t0
    print(f"⏱️  Time: {t_object:.4f}s")
    print(f"   Unique Neighbors: {len(object_results)}")

    # ---------------------------------------------------------
    # B. RAW LOGIC (The "New Way")
    # ---------------------------------------------------------
    print(f"\n[B] Running Raw Logic (Cold Start - Building Matrix)...")
    t0 = time.time()

    # CRITICAL FIX: Pass PRIMITIVES, not Objects
    raw_results_cold = topology.get_raw_successor(input_primitives)

    t_raw_cold = time.time() - t0
    print(f"⏱️  Time: {t_raw_cold:.4f}s")
    print(f"   Count: {len(raw_results_cold)}")

    print(f"\n[B] Running Raw Logic (Warm Start - JAX Cached)...")
    t0 = time.time()

    # Second call uses the cached BCOO matrix
    raw_results_warm = topology.get_raw_successor(input_primitives)

    t_raw_warm = time.time() - t0
    print(f"⏱️  Time: {t_raw_warm:.4f}s")

    # ---------------------------------------------------------
    # C. VERIFICATION
    # ---------------------------------------------------------
    print(f"\n--- Results ---")
    speedup = t_object / t_raw_warm
    print(f"🚀 Speedup (Warm vs Object): {speedup:.1f}x")

    # Correctness Check
    # We must compare apples to apples. Convert Raw Strings -> AbstractStates for check
    raw_results_as_objects = {AbstractState(s, {}) for s in raw_results_warm}

    if raw_results_as_objects == object_results:
        print("✅ SUCCESS: Raw Matrix results match Legacy Object results exactly.")
    else:
        print(f"❌ FAILURE: Mismatch.")
        print(f"   Object Count: {len(object_results)}")
        print(f"   Raw Count:    {len(raw_results_as_objects)}")

        diff = object_results - raw_results_as_objects
        if diff:
            sample = list(diff)[:3]
            print(f"   Missing in Raw: {sample}")


if __name__ == "__main__":
    test_run_million_state_benchmark()