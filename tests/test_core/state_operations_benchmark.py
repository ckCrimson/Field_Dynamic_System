"""
Performance Benchmark for State Space 'Map' Operations.
Compares Manual Iteration vs. Batched Map.
"""
import time
import jax.numpy as jnp
import numpy as np
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace

# Use a large number to make the difference obvious
N_VECTORS = 100_000
N_ABSTRACT = 1000


def benchmark_vector_map():
    print(f"\n--- 1. Vector Map Benchmark (N={N_VECTORS}) ---")

    # 1. Setup: A massive space of 100k vectors
    # (Simulating a scenario where you want to calculate energy for every valid state)
    raw_data = np.random.rand(N_VECTORS, 3)
    vectors = [VectorState(tuple(row)) for row in raw_data]
    space = VectorStateSpace(vectors, dim=3)

    # The Operation: Calculate Distance from Origin
    # Written to be JAX-compatible (handles both single and batch via axis=-1)
    # ... inside benchmark_vector_map ...

    # The Operation: Calculate Distance from Origin
    def op_magnitude(state: VectorState):
        # FIX: We must convert the tuple to an array first!
        # This conversion overhead is exactly what kills performance in loops.
        vec = jnp.array(state.values)
        return jnp.linalg.norm(vec, axis=-1)

    # ... rest of the function remains the same ...

    # 2. Naive Way (User loops manually)
    print("Running Naive Loop...")
    start = time.time()
    # User gets list of objects, loops, calls function, stacks result
    _ = jnp.stack([op_magnitude(v) for v in space.allowed_states])
    naive_time = time.time() - start
    print(f"Naive Loop:   {naive_time:.4f}s")

    # 3. Optimized Map Way (Our new method)
    print("Running Batched Map...")
    # Warmup JAX compilation
    _ = space.map(op_magnitude).block_until_ready()

    start = time.time()
    result = space.map(op_magnitude)
    result.block_until_ready()  # Wait for GPU/CPU to finish
    map_time = time.time() - start

    print(f"Space.map():  {map_time:.4f}s")

    # Stats
    speedup = naive_time / map_time
    print(f"🚀 Speedup: {speedup:.1f}x")

    # Verification
    assert result.shape == (N_VECTORS,)


def benchmark_abstract_map():
    print(f"\n--- 2. Abstract Map Benchmark (N={N_ABSTRACT}) ---")

    # 1. Setup
    states = [AbstractState(f"S_{i}", {"val": i}) for i in range(N_ABSTRACT)]
    space = AbstractDiscreteStateSpace(set(states))

    # The Operation: String manipulation (CPU bound, Python logic)
    def op_describe(state: AbstractState):
        return f"{state.name}_processed"

    # 2. Naive Way
    start = time.time()
    _ = [op_describe(s) for s in space.allowed_states]
    naive_time = time.time() - start
    print(f"Naive Loop:   {naive_time:.6f}s")

    # 3. Map Way (Internal loop)
    start = time.time()
    _ = space.map(op_describe)
    map_time = time.time() - start
    print(f"Space.map():  {map_time:.6f}s")

    print(f"Difference: {naive_time / map_time:.2f}x (Expect ~1.0x)")


if __name__ == "__main__":
    benchmark_vector_map()
    benchmark_abstract_map()