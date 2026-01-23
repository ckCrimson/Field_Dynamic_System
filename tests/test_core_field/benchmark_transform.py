import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.transform import LinearTransform
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransformer
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace


def run_benchmark():
    print("=========================================================")
    print("🚀 BENCHMARK: Field Transform (Vectorized vs Loop)")
    print("=========================================================")

    # -------------------------------------------------
    # 1. SETUP (100,000 Vectors)
    # -------------------------------------------------
    N_STATES = 100_000
    DIM = 2

    print(f"-> Generating {N_STATES} random 2D vectors...")

    # We use AbstractSpace for speed setup (avoiding VectorState object overhead for this test)
    # We cheat and inject the buffer directly to isolate Transform performance
    mock_space = AbstractDiscreteStateSpace(range(N_STATES))
    algebra = VectorFieldAlgebra(dim=DIM)

    # Random Field Data (N, 2)
    key = jax.random.PRNGKey(0)
    random_buffer = jax.random.normal(key, (N_STATES, DIM))

    # Create the Mapper (Input)
    mapper = DiscreteFieldMapper(mock_space, algebra, explicit_buffer=random_buffer)

    # Define Rotation Matrix (Random Angle)
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = jnp.array([[c, -s], [s, c]])  # (2, 2)

    transform_op = LinearTransform(rotation_matrix)

    print(f"-> Setup Complete. Rotation Angle: {theta:.2f} rad")
    print("---------------------------------------------------------")

    # -------------------------------------------------
    # 2. COMPETITOR A: Python Loop (Sequential)
    # -------------------------------------------------
    print("running Sequential Loop (Python)...")

    # Force JAX sync before starting
    random_buffer.block_until_ready()

    start_time = time.perf_counter()

    # Simulation of "Iterating and transforming"
    # We convert to numpy to simulate standard Python overhead
    np_buffer = np.array(random_buffer)
    np_matrix = np.array(rotation_matrix)

    results = []
    for i in range(N_STATES):
        vec = np_buffer[i]
        # v_new = M @ v
        res = np_matrix @ vec
        results.append(res)

    end_time = time.perf_counter()
    time_seq = end_time - start_time
    print(f"⏱️ Sequential Time: {time_seq:.5f} sec")

    # -------------------------------------------------
    # 3. COMPETITOR B: Vectorized (FieldSpaceTransformer)
    # -------------------------------------------------
    print("running Vectorized Transform (JAX)...")

    # Warmup (Compile JIT)
    _ = FieldSpaceTransformer.apply(mapper, transform_op ,VectorFieldAlgebra())

    # Sync and Start Timer
    random_buffer.block_until_ready()
    start_time = time.perf_counter()

    # The Actual Operation
    result_mapper = FieldSpaceTransformer.apply(mapper, transform_op, VectorFieldAlgebra())

    # Force computation to finish
    result_mapper.explicit_buffer.block_until_ready()

    end_time = time.perf_counter()
    time_vec = end_time - start_time
    print(f"⏱️ Vectorized Time: {time_vec:.5f} sec")

    # -------------------------------------------------
    # 4. RESULTS
    # -------------------------------------------------
    print("---------------------------------------------------------")
    speedup = time_seq / time_vec
    print(f"🏆 Speedup Factor: {speedup:.2f}x")

    if speedup > 100:
        print("✅ SUCCESS: Target of 100x speedup achieved.")
    else:
        print("⚠️ WARNING: Speedup target not met.")


if __name__ == "__main__":
    run_benchmark()