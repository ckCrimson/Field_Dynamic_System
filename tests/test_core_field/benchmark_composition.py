import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposer
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace


def run_composition_benchmark():
    print("=========================================================")
    print("🚀 BENCHMARK PART 1: Aligned Field Composition (Fast Path)")
    print("=========================================================")

    # -------------------------------------------------
    # 1. SETUP (100,000 States)
    # -------------------------------------------------
    N_STATES = 100_000
    DIM = 3

    print(f"-> Generating {N_STATES} states & random vectors...")

    # Shared Space (Fast Path trigger)
    space_shared = AbstractDiscreteStateSpace(range(N_STATES))
    algebra = VectorFieldAlgebra(dim=DIM)
    op_add = AdditionComposition()

    # Generate Random Data
    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(2)

    buf1 = jax.random.normal(key1, (N_STATES, DIM))
    buf2 = jax.random.normal(key2, (N_STATES, DIM))

    f1 = DiscreteFieldMapper(space_shared, algebra, explicit_buffer=buf1)
    f2 = DiscreteFieldMapper(space_shared, algebra, explicit_buffer=buf2)

    print("-> Setup Complete.")
    print("---------------------------------------------------------")

    # -------------------------------------------------
    # 2. COMPETITOR A: Python Loop (Sequential)
    # -------------------------------------------------
    print("running Sequential Loop (Naive Addition)...")
    buf1.block_until_ready()

    np_buf1 = np.array(buf1)
    np_buf2 = np.array(buf2)

    start_time = time.perf_counter()

    results = []
    # Simulating: for s in states: val = f1[s] + f2[s]
    for i in range(N_STATES):
        v1 = np_buf1[i]
        v2 = np_buf2[i]
        res = v1 + v2
        results.append(res)

    end_time = time.perf_counter()
    time_seq = end_time - start_time
    print(f"⏱️ Sequential Time: {time_seq:.5f} sec")

    # -------------------------------------------------
    # 3. COMPETITOR B: FieldSpaceComposer (Vectorized)
    # -------------------------------------------------
    print("running FieldSpaceComposer (Aligned)...")

    # Warmup
    _ = FieldSpaceComposer.compose(f1, f2, op_add, algebra)
    buf1.block_until_ready()

    start_time = time.perf_counter()

    f3 = FieldSpaceComposer.compose(f1, f2, op_add, algebra)
    f3.explicit_buffer.block_until_ready()

    end_time = time.perf_counter()
    time_vec = end_time - start_time
    print(f"⏱️ Vectorized Time: {time_vec:.5f} sec")

    speedup = time_seq / time_vec
    print(f"🏆 Speedup Factor: {speedup:.2f}x")

    # =========================================================
    # PART 2: UNALIGNED SPACES (Symbolic / Object Overhead)
    # =========================================================
    print("\n=========================================================")
    print("🚀 BENCHMARK PART 2: Unaligned Spaces (Symbolic Join)")
    print("=========================================================")

    # Scenario:
    # Space A: [0 ... 50,000]
    # Space B: [25,000 ... 75,000]
    # Intersection: 25k. Union: 75k.

    SIZE_A = 50_000
    OFFSET_B = 25_000
    SIZE_B = 50_000

    range_a = range(0, SIZE_A)
    range_b = range(OFFSET_B, OFFSET_B + SIZE_B)

    print(f"-> Generating Unaligned Spaces (Overlap 50%)...")
    space_a = AbstractDiscreteStateSpace(range_a)
    space_b = AbstractDiscreteStateSpace(range_b)

    # Buffers
    key3 = jax.random.PRNGKey(3)
    key4 = jax.random.PRNGKey(4)
    buf_a = jax.random.normal(key3, (SIZE_A, DIM))
    buf_b = jax.random.normal(key4, (SIZE_B, DIM))

    # Unaligned Mappers
    field_a = DiscreteFieldMapper(space_a, algebra, explicit_buffer=buf_a)
    field_b = DiscreteFieldMapper(space_b, algebra, explicit_buffer=buf_b)

    print("-> Setup Complete.")
    print("---------------------------------------------------------")

    # -------------------------------------------------
    # 4. COMPETITOR A: Python Dictionary Join (Naive)
    # -------------------------------------------------
    print("running Python Dict Join (Hash Map)...")

    list_states_a = list(range_a)
    list_states_b = list(range_b)
    arr_a = np.array(buf_a)
    arr_b = np.array(buf_b)

    start_time = time.perf_counter()

    # Naive Logic: Build Dict, Sum collisions
    merged_data = {}
    for i, s in enumerate(list_states_a):
        merged_data[s] = arr_a[i]
    for i, s in enumerate(list_states_b):
        if s in merged_data:
            merged_data[s] += arr_b[i]
        else:
            merged_data[s] = arr_b[i]

    final_res_py = list(merged_data.values())

    end_time = time.perf_counter()
    time_seq_unaligned = end_time - start_time
    print(f"⏱️ Python Dict Time: {time_seq_unaligned:.5f} sec")

    # -------------------------------------------------
    # 5. COMPETITOR B: FieldSpaceComposer (Symbolic Wrapper)
    # -------------------------------------------------
    print("running FieldSpaceComposer.compose (Object Wrapper)...")

    # Warmup
    _ = FieldSpaceComposer.compose(field_a, field_b, op_add, algebra)

    start_time = time.perf_counter()

    # This includes Python overhead (Set intersections etc.)
    f_res_unaligned = FieldSpaceComposer.compose(field_a, field_b, op_add, algebra)
    f_res_unaligned.explicit_buffer.block_until_ready()

    end_time = time.perf_counter()
    time_obj_unaligned = end_time - start_time
    print(f"⏱️ Object Wrapper Time: {time_obj_unaligned:.5f} sec")

    # =========================================================
    # PART 3: RAW SPARSE KERNEL (The Simulation Engine)
    # =========================================================
    print("\n=========================================================")
    print("🚀 BENCHMARK PART 3: Raw Sparse Kernel (compose_raw)")
    print("=========================================================")
    print("Simulating the 'Sparse Loop' (Merging ID lists)...")

    # Input: Raw Integer IDs and Values
    # We pretend we already mapped states to integers (0..50k and 25k..75k)
    ids_a_raw = jnp.arange(0, SIZE_A)
    ids_b_raw = jnp.arange(OFFSET_B, OFFSET_B + SIZE_B)

    # Ensure on device
    ids_a_raw.block_until_ready()
    ids_b_raw.block_until_ready()
    buf_a.block_until_ready()
    buf_b.block_until_ready()

    print("running FieldSpaceComposer.compose_raw...")

    # Warmup
    _ = FieldSpaceComposer.compose_raw(
        ids_a_raw, buf_a,
        ids_b_raw, buf_b,
        op_add
    )

    start_time = time.perf_counter()

    # THIS is what runs in your loop:
    # (IDs, Vals) + (IDs, Vals) -> (MergedIDs, MergedVals)
    unique_ids, merged_vals = FieldSpaceComposer.compose_raw(
        ids_a_raw, buf_a,
        ids_b_raw, buf_b,
        op_add
    )
    merged_vals.block_until_ready()

    end_time = time.perf_counter()
    time_raw = end_time - start_time

    print(f"⏱️ JAX Sparse Kernel Time: {time_raw:.5f} sec")

    # Compare against Python Dict (Part 2)
    speedup_raw = time_seq_unaligned / time_raw
    print(f"🏆 Speedup Factor (vs Dict): {speedup_raw:.2f}x")

    # Verification
    # 1. Size Check
    expected_size = 75_000
    if unique_ids.shape[0] == expected_size:
        print(f"✅ Size Verified: {unique_ids.shape[0]}")
    else:
        print(f"❌ ERROR: Size {unique_ids.shape[0]}, expected {expected_size}")

    # 2. Sum Check
    sum_py = np.sum(np.array(final_res_py))
    sum_jax = jnp.sum(merged_vals)

    if np.isclose(sum_py, sum_jax, rtol=1e-3):
        print("✅ Sum Verified")
    else:
        print(f"❌ ERROR: Sum Mismatch. Py: {sum_py}, JAX: {sum_jax}")


if __name__ == "__main__":
    run_composition_benchmark()