import time
import jax.numpy as jnp
import numpy as np
import pytest
from abc import ABC, abstractmethod

from src.field_dynamic_system.core.field.compositions import FieldComposition
# Adjust import to point to your actual file
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposer


# =========================================================
# 0. DEFINITIONS (Mocking the classes for the test file)
# =========================================================


class AdditionComposition(FieldComposition):
    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a + b

    def get_identity(self, shape: tuple, dtype=jnp.float32) -> jnp.ndarray:
        return jnp.zeros(shape, dtype=dtype)


# =========================================================
# 1. LOGIC TEST (Small Scale - 2 States)
# =========================================================

def test_batch_logic_small():
    print("\n=========================================================")
    print("🧪 TEST: Batch Logic (Overlap & Collision)")
    print("=========================================================")

    # SCENARIO:
    # We have 3 Fields and 2 States (ID=0, ID=1).
    # Field A: {State 0: 10.0}
    # Field B: {State 0:  5.0, State 1: 2.0}
    # Field C: {State 1:  3.0}

    # EXPECTED RESULT (Superposition):
    # State 0: 10 + 5 = 15.0
    # State 1:  2 + 3 =  5.0

    op = AdditionComposition()

    # 1. Define Inputs (Sparse Format)
    # Field A (State 0)
    ids_a = jnp.array([[0]], dtype=jnp.int32)
    vals_a = jnp.array([[10.0]], dtype=jnp.float32)

    # Field B (State 0 and 1)
    ids_b = jnp.array([[0], [1]], dtype=jnp.int32)
    vals_b = jnp.array([[5.0], [2.0]], dtype=jnp.float32)

    # Field C (State 1)
    ids_c = jnp.array([[1]], dtype=jnp.int32)
    vals_c = jnp.array([[3.0]], dtype=jnp.float32)

    batch_inputs = [
        (ids_a, vals_a),
        (ids_b, vals_b),
        (ids_c, vals_c)
    ]

    # 2. RUN BATCH COMPOSE
    print(f"-> Composing {len(batch_inputs)} sparse fields...")
    unique_ids, unique_vals = FieldSpaceComposer.compose_batch(batch_inputs, op)

    # 3. VERIFY
    # Flatten for easier reading
    u_ids = unique_ids.flatten()
    u_vals = unique_vals.flatten()

    print(f"   Result IDs:   {u_ids}")
    print(f"   Result Vals:  {u_vals}")

    # Locate State 0
    # Note: jnp.unique sorts the output, so 0 should be first, but we use where() to be safe.
    idx_0 = int(jnp.where(u_ids == 0)[0][0])
    val_0 = float(u_vals[idx_0])
    print(f"   State 0 Sum: {val_0} (Expected 15.0)")

    # Locate State 1
    idx_1 = int(jnp.where(u_ids == 1)[0][0])
    val_1 = float(u_vals[idx_1])
    print(f"   State 1 Sum: {val_1} (Expected 5.0)")

    assert val_0 == 15.0
    assert val_1 == 5.0
    assert unique_ids.shape[0] == 2  # Only 2 unique states exist

    print("✅ SUCCESS: Logic Test Passed.")


# =========================================================
# 2. BENCHMARK (1,000 Fields)
# =========================================================

def test_batch_benchmark_large():
    print("\n=========================================================")
    print("🚀 BENCHMARK: 1,000 Field Composition")
    print("=========================================================")

    N_FIELDS = 1000
    op = AdditionComposition()

    # 1. GENERATE DATA
    # 1,000 fields. Each has exactly 1 state.
    # Field i has State i with Value 1.0.
    # Initially: No collisions, just massive aggregation.
    print(f"-> Generating {N_FIELDS} unique sparse fields...")

    batch_inputs = []
    for i in range(N_FIELDS):
        ids = jnp.array([[i]], dtype=jnp.int32)
        vals = jnp.array([[1.0]], dtype=jnp.float32)
        batch_inputs.append((ids, vals))

    # 2. RUN BENCHMARK (Aggregation)
    print("-> Running compose_batch (Aggregation)...")
    t0 = time.time()

    res_ids, res_vals = FieldSpaceComposer.compose_batch(batch_inputs, op)

    # Force Wait
    res_vals.block_until_ready()

    t_batch = time.time() - t0
    print(f"⏱️  Time: {t_batch:.4f}s")

    # Verify
    assert res_ids.shape[0] == N_FIELDS, "Should have 1,000 unique states"
    assert res_vals.shape[0] == N_FIELDS

    # 3. RUN HEAVY COLLISION (Merge with Self)
    # Merging the list with itself = 2,000 fields.
    # Every ID appears twice.
    # This forces the kernel to perform massive scatter-add operations.
    print("\n-> Testing Heavy Collision (2,000 overlapping fields)...")

    heavy_batch = batch_inputs + batch_inputs  # Python List concat

    t0 = time.time()
    col_ids, col_vals = FieldSpaceComposer.compose_batch(heavy_batch, op)
    col_vals.block_until_ready()
    t_col = time.time() - t0

    print(f"⏱️  Time: {t_col:.4f}s")

    # Verification
    # Still 1,000 unique IDs
    # But values should be 2.0 (1.0 + 1.0)
    assert col_ids.shape[0] == N_FIELDS
    assert col_vals[0][0] == 2.0

    print("✅ SUCCESS: Benchmark Passed.")


if __name__ == "__main__":
    test_batch_logic_small()
    test_batch_benchmark_large()