"""
Performance Benchmark for State Space 'Map' Operations.
Compares Manual Iteration vs. Batched Map.
"""
import time
import jax.numpy as jnp
import numpy as np
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace

# 100k is enough to see the difference
N_VECTORS = 100_000
N_ABSTRACT = 1000

def benchmark_vector_map():
    print(f"\n--- 1. Vector Map Benchmark (N={N_VECTORS}) ---")

    # 1. Setup
    raw_data = np.random.rand(N_VECTORS, 3)
    vectors = [VectorState(tuple(row)) for row in raw_data]
    space = VectorStateSpace(vectors, dim=3)

    # --- THE CRITICAL FIX ---
    def op_magnitude(state):
        # CASE A: Naive Loop (state is VectorState Object)
        if hasattr(state, 'values'):
            # MUST convert tuple -> jnp.array manually
            vec = jnp.array(state.values)

        # CASE B: Optimized Map (state is already a JAX Array)
        else:
            vec = state

        return jnp.linalg.norm(vec, axis=-1)
    # -------------------------

    # 2. Naive Way (User loops manually)
    print("Running Naive Loop...")
    start = time.time()
    # Explicit loop over objects
    _ = jnp.stack([op_magnitude(v) for v in space.states])
    naive_time = time.time() - start
    print(f"Naive Loop:   {naive_time:.4f}s")

    # 3. Optimized Map Way
    print("Running Batched Map...")
    if hasattr(space, 'map'):
        # Warmup
        try:
            _ = space.map(op_magnitude).block_until_ready()

            start = time.time()
            result = space.map(op_magnitude)
            result.block_until_ready()
            map_time = time.time() - start

            print(f"Space.map():  {map_time:.4f}s")
            print(f"🚀 Speedup:   {naive_time / map_time:.1f}x")
        except Exception as e:
            print(f"❌ Map Failed: {e}")
            map_time = 1.0
    else:
        print("⚠️ Space.map() not implemented.")


def benchmark_abstract_map():
    print(f"\n--- 2. Abstract Map Benchmark (N={N_ABSTRACT}) ---")

    states = [AbstractState(f"S_{i}", {"val": i}) for i in range(N_ABSTRACT)]
    space = AbstractDiscreteStateSpace(set(states))

    def op_describe(state: AbstractState):
        return f"{state.name}_processed"

    # Naive
    start = time.time()
    _ = [op_describe(s) for s in space.states]
    naive_time = time.time() - start
    print(f"Naive Loop:   {naive_time:.6f}s")

    # Map
    if hasattr(space, 'map'):
        start = time.time()
        # This returns a LIST now (because strings can't be stacked)
        _ = space.map(op_describe)
        map_time = time.time() - start
        print(f"Space.map():  {map_time:.6f}s")
        print(f"Difference:   {naive_time / map_time:.2f}x")


if __name__ == "__main__":
    benchmark_vector_map()
    benchmark_abstract_map()