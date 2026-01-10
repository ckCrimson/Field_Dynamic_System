"""
Benchmark: Discrete State Transformations.
Compares 'Naive Python Loop' vs 'Transformation Pipeline'.
"""
import time
import jax.numpy as jnp
import numpy as np
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state.transformation import DiscreteStateTransformation

N_VECTORS = 50_000
N_ABSTRACT = 5_000


def benchmark_vector_transform():
    print(f"\n--- 1. Vector Transformation (N={N_VECTORS}) ---")

    # 1. Setup: 50k vectors
    # Scenario: Normalize vectors (Math heavy operation)
    raw_data = np.random.rand(N_VECTORS, 3).astype(np.float32)
    vectors = [VectorState(tuple(row)) for row in raw_data]
    space = VectorStateSpace(vectors, dim=3)

    # The Operation: Normalize vector
    def op_normalize(v):
        # FIX: Extract data from the Object wrapper
        if hasattr(v, 'values'):
            # v is a VectorState (either single or batched)
            # We must convert the tuple/array inside it to a JAX array
            arr = jnp.array(v.values)
        elif isinstance(v, (jnp.ndarray, np.ndarray)):
            # v is already raw data (e.g. from a raw JAX map)
            arr = v
        else:
            # v is a tuple or list
            arr = jnp.array(v)

        # Add epsilon to avoid div by zero
        norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / (norm + 1e-6)

    # --- A. Naive Python Approach ---
    print("Running Naive Loop...")
    start = time.time()

    # We must loop, convert to JAX, compute, convert back
    new_vectors_naive = []
    for vec in space.allowed_vectors:
        # High overhead here: dispatching JAX kernel 50,000 times
        res = op_normalize(jnp.array(vec.values))
        new_vectors_naive.append(VectorState(tuple(res.tolist())))

    _ = VectorStateSpace(new_vectors_naive, dim=3)

    naive_time = time.time() - start
    print(f"Naive Time:      {naive_time:.4f}s")

    # --- B. Transformation Pipeline ---
    print("Running Pipeline...")
    transformer = DiscreteStateTransformation(
        operation=op_normalize,
        target_class=VectorStateSpace
    )

    # Warmup JIT
    # We run on a small slice to trigger compilation without waiting too long
    small_space = VectorStateSpace(vectors[:10], dim=3)
    _ = transformer.transform(small_space)

    start = time.time()
    _ = transformer.transform(space)
    pipe_time = time.time() - start

    print(f"Pipeline Time:   {pipe_time:.4f}s")
    print(f"🚀 Speedup:      {naive_time / pipe_time:.1f}x")


def benchmark_abstract_transform():
    print(f"\n--- 2. Abstract Transformation (N={N_ABSTRACT}) ---")

    # 1. Setup
    states = {AbstractState(f"S_{i}", {}) for i in range(N_ABSTRACT)}
    space = AbstractDiscreteStateSpace(states)

    # Operation: Rename
    def op_rename(state):
        return f"New_{state.name}"

    # --- A. Naive Loop ---
    start = time.time()
    new_states_a = {AbstractState(op_rename(s), {}) for s in space.allowed_states}
    _ = AbstractDiscreteStateSpace(new_states_a)
    naive_time = time.time() - start
    print(f"Naive Time:      {naive_time:.4f}s")

    # --- B. Pipeline ---
    transformer = DiscreteStateTransformation(
        operation=op_rename,
        target_class=AbstractDiscreteStateSpace
    )

    start = time.time()
    _ = transformer.transform(space)
    pipe_time = time.time() - start

    print(f"Pipeline Time:   {pipe_time:.4f}s")
    print(f"Speedup:         {naive_time / pipe_time:.2f}x (Expect ~1.0x)")


if __name__ == "__main__":
    benchmark_vector_transform()
    benchmark_abstract_transform()