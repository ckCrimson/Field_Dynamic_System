"""
Performance Benchmark for Discrete State Spaces.
"""
import time
import jax.numpy as jnp
import numpy as np  # Used for random generation
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, VectorStateSpace

N_ITEMS = 100_000


def benchmark_abstract_lookup():
    print(f"\n--- Benchmarking Abstract Lookup (N={N_ITEMS}) ---")

    # 1. Setup Universe of 100 states
    states = [AbstractState(f"S_{i}", {}) for i in range(100)]
    space = AbstractDiscreteStateSpace(set(states))  # All are valid

    # Generate test batch (mixed valid/invalid)
    test_batch_objs = [states[i % 100] for i in range(N_ITEMS)]

    # 2. Python Set Lookup (Cold Path - Object Validation)
    start = time.time()
    _ = space.contains(test_batch_objs)
    cold_time = time.time() - start
    print(f"Object Lookup (Python Set): {cold_time:.4f}s")

    # 3. JAX ID Lookup (Hot Path - Simulation Loop)
    # Checking IDs is just (0 <= x < N). Should be instant.
    test_batch_ids = jnp.arange(N_ITEMS) % 150  # Some valid (0-99), some invalid (100-149)

    # Warmup JAX
    _ = space.contains(test_batch_ids).block_until_ready()

    start = time.time()
    _ = space.contains(test_batch_ids).block_until_ready()
    hot_time = time.time() - start

    print(f"ID Lookup (JAX Array):      {hot_time:.6f}s")
    print(f"🚀 Speedup Factor: {cold_time / hot_time:.1f}x")


def benchmark_vector_broadcast():
    print(f"\n--- Benchmarking Vector Broadcasting (N={N_ITEMS}) ---")

    # 1. Setup: A space with 6 valid directions (Cube faces)
    dirs = [
        VectorState((1, 0, 0)), VectorState((-1, 0, 0)),
        VectorState((0, 1, 0)), VectorState((0, -1, 0)),
        VectorState((0, 0, 1)), VectorState((0, 0, -1))
    ]
    space = VectorStateSpace(dirs, dim=3)

    # 2. Generate 100k random vectors
    # We use numpy for generation speed, then convert to JAX
    raw_vecs = np.random.rand(N_ITEMS, 3).astype(np.float32)
    # Ensure some match exactly by overwriting
    raw_vecs[0:1000] = [1, 0, 0]

    jax_vecs = jnp.array(raw_vecs)

    # Warmup
    _ = space.contains(jax_vecs).block_until_ready()

    # 3. Measure Broadcasting Speed
    # Logic: Compares (100k, 1, 3) against (1, 6, 3) -> Creates (100k, 6) bool matrix
    start = time.time()
    mask = space.contains(jax_vecs).block_until_ready()
    duration = time.time() - start

    print(f"Vector Broadcast Check: {duration:.4f}s")
    print(f"Throughput: {N_ITEMS / duration / 1e6:.2f} Million vectors/sec")


if __name__ == "__main__":
    benchmark_abstract_lookup()
    benchmark_vector_broadcast()