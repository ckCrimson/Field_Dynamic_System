"""
Benchmark: Discrete State Transformations.
Compares 'Naive Python Loop' vs 'Transformation Pipeline'.
"""
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state.transformation import DiscreteStateTransformation, AbstractStateTransformation, \
    VectorStateTransformation

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
    transformer = VectorStateTransformation(
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
    transformer = AbstractStateTransformation(
        operation=op_rename,
        target_class=AbstractDiscreteStateSpace
    )

    start = time.time()
    _ = transformer.transform(space)
    pipe_time = time.time() - start

    print(f"Pipeline Time:   {pipe_time:.4f}s")
    print(f"Speedup:         {naive_time / pipe_time:.2f}x (Expect ~1.0x)")


def test_benchmark_vector_to_abstract(capsys):
    """
    Benchmark converting Geometry (Vector) -> Semantics (Abstract).
    Corrected to ensure N=100,000 unique items and handle JAX warmup.
    """
    N = 1_000_000  # Was 100,000
    print(f"\n\n--- Conversion Benchmark (N={N}) ---")

    # --- 0. JAX WARMUP (Crucial!) ---
    # We run a dummy calculation so the library loads BEFORE we start the timer.
    print("Warming up JAX...")
    _ = jnp.linalg.norm(jnp.array([[1.0, 1.0]]))

    # --- 1. Setup Unique Data ---
    # We add 'i' to the y-axis so every vector is unique.
    # Logic:
    #   if i%3==0 (IN):  x=0.5, y=tiny_shift -> Norm < 1
    #   if i%3==1 (ON):  x=1.0, y=0.0        -> Norm = 1
    #   if i%3==2 (OUT): x=2.0, y=tiny_shift -> Norm > 1
    raw_vectors = []
    for i in range(N):
        if i % 3 == 0:
            vec = (0.5, i * 1e-10)  # Unique, effectively inside
        elif i % 3 == 1:
            vec = (1.0, 0.0)  # Exactly 1.0 (Keep duplicates here to test ON)
        else:
            vec = (2.0, i * 1e-10)  # Unique, effectively outside
        raw_vectors.append(VectorState(vec))

    print("Building Vector Space (collapsing duplicates)...")
    vector_space = VectorStateSpace(raw_vectors, dim=2)

    real_N = vector_space.num_states
    print(f"Actual Unique States to Process: {real_N}")
    # This should now be close to ~66,000 (since the ON ones might collapse, but IN/OUT are unique)

    # --- APPROACH 1: NAIVE PYTHON LOOP ---
    print("Running Naive Python Loop...")
    start = time.time()

    naive_results = set()
    for v in vector_space.allowed_states:
        x, y = v.values
        norm = math.sqrt(x ** 2 + y ** 2)

        if math.isclose(norm, 1.0, rel_tol=1e-5):
            label = "ON"
        elif norm < 1.0:
            label = "IN"
        else:
            label = "OUT"

        naive_results.add(AbstractState(label, {}))

    naive_space = AbstractDiscreteStateSpace(naive_results)
    naive_dur = time.time() - start

    # --- APPROACH 2: HYBRID JAX BATCHING ---
    print("Running Hybrid JAX Batching (JIT Compiled)...")

    # 1. Define the logic as a pure function
    # The @jax.jit decorator compiles this into a single C++ kernel!
    print("Running Hybrid JAX Batching (JIT Compiled)...")

    # 1. Define the Math Kernel (Must return fixed shapes)
    print("Running Hybrid JAX Batching (JIT Compiled)...")

    # 1. Define the Math Kernel (Must return fixed shapes)
    @jax.jit
    def calculate_categories(matrix):
        """
        Calculates categories for ALL pixels in parallel.
        Returns an array of shape (N,) containing integers 0, 1, or 2.
        """
        norms = jnp.linalg.norm(matrix, axis=1)
        tol = 1e-5

        # Default to OUT (2)
        cats = jnp.full(norms.shape, 2, dtype=jnp.int32)
        # IN (0)
        cats = jnp.where(norms < (1.0 - tol), 0, cats)
        # ON (1)
        cats = jnp.where(jnp.abs(norms - 1.0) < tol, 1, cats)

        return cats

    # 2. Warmup Compilation
    print("Compiling JAX Kernel...")
    matrix = vector_space.get_matrix()
    # Trigger compilation on the math kernel
    _ = calculate_categories(matrix).block_until_ready()

    # 3. Timed Run
    start = time.time()

    # Step A: Run the JIT-ed Math (Fast!)
    all_categories = calculate_categories(matrix)
    all_categories.block_until_ready()  # Ensure GPU/CPU is done

    # Step B: Find Unique Values (Run in Eager Mode outside JIT)
    # This is allowed because we are back in Python/Eager JAX land
    unique_cats_jax = jnp.unique(all_categories)

    # Step C: Convert to Python
    unique_cats_list = unique_cats_jax.tolist()

    labels = ("IN", "ON", "OUT")
    hybrid_results = {AbstractState(labels[i], {}) for i in unique_cats_list}

    hybrid_space = AbstractDiscreteStateSpace(hybrid_results)
    hybrid_dur = time.time() - start
    # --- REPORT ---
    print(f"Naive Time:   {naive_dur:.4f}s")
    print(f"Hybrid Time:  {hybrid_dur:.4f}s")

    speedup = naive_dur / hybrid_dur if hybrid_dur > 0 else 0
    print(f"🚀 Speedup:     {speedup:.1f}x")

    assert naive_space.num_states == hybrid_space.num_states

if __name__ == "__main__":
    #benchmark_vector_transform()
    #benchmark_abstract_transform()
    test_benchmark_vector_to_abstract()