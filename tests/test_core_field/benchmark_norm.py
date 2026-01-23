import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra, RealFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.transform import NormTransform
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransformer
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace


def run_norm_benchmark():
    print("=========================================================")
    print("🚀 BENCHMARK: Norm Transform (Vector -> Scalar)")
    print("=========================================================")

    # -------------------------------------------------
    # 1. SETUP (100,000 3D Vectors)
    # -------------------------------------------------
    N_STATES = 100_0000
    DIM = 3

    print(f"-> Generating {N_STATES} random 3D vectors...")

    mock_space = AbstractDiscreteStateSpace(range(N_STATES))
    algebra_vec = VectorFieldAlgebra(dim=DIM)

    # Random Field Data (N, 3)
    key = jax.random.PRNGKey(42)
    random_buffer = jax.random.normal(key, (N_STATES, DIM))

    mapper_vec = DiscreteFieldMapper(mock_space, algebra_vec, explicit_buffer=random_buffer)

    # The Operator
    transform_op = NormTransform()

    print(f"-> Input Algebra: {type(mapper_vec.algebra).__name__}")
    print("---------------------------------------------------------")

    # -------------------------------------------------
    # 2. COMPETITOR A: Python Loop (Sequential Norm)
    # -------------------------------------------------
    print("running Sequential Loop (numpy.linalg.norm)...")
    random_buffer.block_until_ready()

    np_buffer = np.array(random_buffer)

    start_time = time.perf_counter()

    results = []
    # Simulating iterating states and computing magnitude
    for i in range(N_STATES):
        vec = np_buffer[i]
        mag = np.linalg.norm(vec)
        results.append(mag)

    end_time = time.perf_counter()
    time_seq = end_time - start_time
    print(f"⏱️ Sequential Time: {time_seq:.5f} sec")

    # -------------------------------------------------
    # 3. COMPETITOR B: Vectorized JAX (FieldSpaceTransformer)
    # -------------------------------------------------
    print("running Vectorized Transform (JAX)...")

    # Warmup
    _ = FieldSpaceTransformer.apply(mapper_vec, transform_op,algebra_vec)
    random_buffer.block_until_ready()

    start_time = time.perf_counter()

    # The Actual Transformation
    mapper_scalar = FieldSpaceTransformer.apply(mapper_vec, transform_op,RealFieldAlgebra())

    mapper_scalar.explicit_buffer.block_until_ready()

    end_time = time.perf_counter()
    time_vec = end_time - start_time
    print(f"⏱️ Vectorized Time: {time_vec:.5f} sec")

    # -------------------------------------------------
    # 4. VERIFICATION & RESULTS
    # -------------------------------------------------
    print("---------------------------------------------------------")

    # Check 1: Speedup
    speedup = time_seq / time_vec
    print(f"🏆 Speedup Factor: {speedup:.2f}x")

    # Check 2: Correctness (First Element)
    # JAX returns (N, 1), Numpy loop returned list of scalars
    val_jax = mapper_scalar.explicit_buffer[0][0]
    val_seq = results[0]

    print(f"🔍 Value Check (State 0):")
    print(f"   Seq: {val_seq:.6f}")
    print(f"   JAX: {val_jax:.6f}")

    if not np.isclose(val_seq, val_jax):
        print("❌ ERROR: Mismatch in calculation values.")

    # Check 3: Algebra Switching
    out_alg_type = type(mapper_scalar.algebra)
    print(f"🧠 Output Algebra: {out_alg_type.__name__}")

    if out_alg_type == RealFieldAlgebra:
        print("✅ SUCCESS: Correctly switched to RealFieldAlgebra.")
    else:
        print(f"❌ ERROR: Expected RealFieldAlgebra, got {out_alg_type}")


if __name__ == "__main__":
    run_norm_benchmark()