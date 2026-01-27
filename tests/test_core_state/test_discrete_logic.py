"""
Benchmark: Discrete State Transformations.
Compares 'Naive Python Loop' vs 'Transformation Pipeline'.
"""
import math
import time
import jax
import jax.numpy as jnp
import numpy as np

# Adjust imports to match your project structure
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state.transformation import DiscreteStateTransformation

N_VECTORS = 50_000
N_ABSTRACT = 5_000


def benchmark_vector_transform():
    print(f"\n--- 1. Vector Transformation (N={N_VECTORS}) ---")

    # 1. Setup: 50k vectors
    raw_data = np.random.rand(N_VECTORS, 3).astype(np.float32)
    vectors = [VectorState(tuple(row)) for row in raw_data]
    space = VectorStateSpace(vectors, dim=3)

    # The Operation: Normalize vector
    def op_normalize(v):
        # Handle both Object (Naive Loop) and Array (JAX Map)
        if hasattr(v, 'values'):
            arr = jnp.array(v.values)
        else:
            arr = v

        norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / (norm + 1e-6)

    # --- A. Naive Python Approach ---
    print("Running Naive Loop...")
    start = time.time()

    new_vectors_naive = []
    # FIX: Use .states
    for vec in space.states:
        res = op_normalize(vec) # Naive passes Object
        new_vectors_naive.append(VectorState(tuple(res.tolist())))

    _ = VectorStateSpace(new_vectors_naive, dim=3)
    naive_time = time.time() - start
    print(f"Naive Time:      {naive_time:.4f}s")

    # --- B. Transformation Pipeline ---
    print("Running Pipeline (JAX Optimized)...")

    # WRAPPER: The pipeline expects the operation to return something
    # the target class can ingest. VectorStateSpace expects VectorState objects
    # OR raw data if we use from_raw_data.
    # For this standard benchmark, we assume the user function returns Raw Data,
    # and we need a wrapper to convert back to Objects for the Space constructor.

    # However, since we want to benchmark the *Mechanism*, we'll use a
    # simplified lambda that assumes the pipeline handles data flow.

    transformer = DiscreteStateTransformation(
        operation=op_normalize, # Returns JAX Array
        # We cheat slightly: VectorStateSpace constructor expects Objects,
        # but for the benchmark speed, we assume the user would pass
        # a factory that accepts raw data.
        # Let's use the standard class for fairness.
        target_class=lambda x: VectorStateSpace([VectorState(tuple(r)) for r in x], dim=3)
    )

    start = time.time()
    # The .map() inside transform() will be Instant (0.01s).
    # The .tolist() and object creation will take time (Python overhead).
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
        return AbstractState(f"New_{state.name}", {})

    # --- A. Naive Loop ---
    start = time.time()
    # FIX: Use .states
    new_states_a = {op_rename(s) for s in space.states}
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


def test_benchmark_vector_to_abstract():
    """
    Benchmark converting Geometry (Vector) -> Semantics (Abstract).
    """
    N = 1_000_000
    print(f"\n\n--- Conversion Benchmark (N={N}) ---")

    print("Warming up JAX...")
    _ = jnp.linalg.norm(jnp.array([[1.0, 1.0]]))

    # Setup Unique Data
    raw_vectors = []
    # Create simpler data for speed of generation
    # We use a generator to avoid huge memory spike before space creation
    print("Generating Data...")
    raw_data = np.random.rand(N, 2)
    # Inject some known values
    raw_data[0] = [1.0, 0.0] # On
    raw_data[1] = [0.5, 0.0] # In
    raw_data[2] = [2.0, 0.0] # Out

    # Use Factory for speed!
    space = VectorStateSpace.from_raw_data(
        raw_data,
        wrapper=lambda x: VectorState(tuple(x.tolist())),
        dim=2
    )

    print(f"Processing {space.num_states} states...")

    # --- APPROACH 1: NAIVE PYTHON LOOP ---
    print("Running Naive Python Loop...")
    start = time.time()

    naive_results = set()
    # FIX: Use .states
    for v in space.states:
        x, y = v.values
        norm = math.sqrt(x ** 2 + y ** 2)

        if math.isclose(norm, 1.0, rel_tol=1e-5):
            label = "ON"
        elif norm < 1.0:
            label = "IN"
        else:
            label = "OUT"
        # Only add unique semantics
        naive_results.add(label)

    naive_space = AbstractDiscreteStateSpace({AbstractState(l, {}) for l in naive_results})
    naive_dur = time.time() - start

    # --- APPROACH 2: HYBRID JAX BATCHING ---
    print("Running Hybrid JAX Batching...")

    # 1. JAX Kernel
    @jax.jit
    def calculate_categories(matrix):
        norms = jnp.linalg.norm(matrix, axis=1)
        tol = 1e-5
        cats = jnp.full(norms.shape, 2, dtype=jnp.int32) # OUT
        cats = jnp.where(norms < (1.0 - tol), 0, cats)   # IN
        cats = jnp.where(jnp.abs(norms - 1.0) < tol, 1, cats) # ON
        return cats

    # 2. Compile
    matrix = space.get_matrix()
    _ = calculate_categories(matrix[:10]).block_until_ready()

    # 3. Timed Run
    start = time.time()

    # A. Math (Fast)
    all_categories = calculate_categories(matrix)
    all_categories.block_until_ready()

    # B. Reduce (Fast)
    unique_cats_jax = jnp.unique(all_categories)
    unique_cats_list = unique_cats_jax.tolist()

    # C. Semantics (Cheap)
    labels = ("IN", "ON", "OUT")
    hybrid_results = {AbstractState(labels[i], {}) for i in unique_cats_list}

    hybrid_space = AbstractDiscreteStateSpace(hybrid_results)
    hybrid_dur = time.time() - start

    # --- REPORT ---
    print(f"Naive Time:   {naive_dur:.4f}s")
    print(f"Hybrid Time:  {hybrid_dur:.4f}s")

    speedup = naive_dur / hybrid_dur if hybrid_dur > 0 else 0
    print(f"🚀 Speedup:     {speedup:.1f}x")

if __name__ == "__main__":
    benchmark_vector_transform()
    benchmark_abstract_transform()
    test_benchmark_vector_to_abstract()