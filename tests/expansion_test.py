import time
import jax.numpy as jnp
from jax import config

from src.field_dynamic_system.core import VectorState

# Enable 64-bit
config.update("jax_enable_x64", True)

# --- IMPORTS ---
from src.field_dynamic_system.core.state.discrete import VectorStateSpace
from src.field_dynamic_system.neighbor.discrete import DeltaTopology
from src.field_dynamic_system.core.field.mappings import FieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.core.field.stratergies import (RealAddition)
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposition, FieldSpaceTransform
from src.field_dynamic_system.core.field.transform import FieldTransform

import time
import jax.numpy as jnp
from jax import config

# Enable 64-bit precision
config.update("jax_enable_x64", True)



# --- UTILS ---
def synchronize(obj):
    """
    BENCHMARK HELPER ONLY.
    Forces JAX to finish computation so the stopwatch is accurate.
    """
    if hasattr(obj, 'block_until_ready'):
        obj.block_until_ready()
    elif hasattr(obj, 'explicit_buffer'):
        obj.explicit_buffer.block_until_ready()
    elif hasattr(obj, '_matrix'):
        obj._matrix.block_until_ready()
    return obj


class ScaleTransform(FieldTransform):
    def __init__(self, scale): self.scale = scale

    def transform(self, x): return x * self.scale


# --- BENCHMARK ---

def run_benchmark():
    print("\n=== BENCHMARK: Expansion Test (Corrected) ===")

    # 1. SETUP: Grid 100x100
    print("-> Generating 10,000 VectorStates...")
    vectors = [VectorState((float(x), float(y))) for y in range(100) for x in range(100)]
    space = VectorStateSpace(vectors, dim=2)

    deltas = [
        (0, 1), (1, 0), (0, 0), (1, 1),
        (-1, 0), (0, -1), (-1, -1), (1, -1)
    ]
    topology = DeltaTopology(space, deltas)

    print(f"-> Space: {space.num_states} states")

    # ==========================================
    # A. TOPOLOGY: Clean User Experience
    # ==========================================
    print("\n[A] Topology Expansion (60 steps)")

    start_v = VectorState((50.0, 50.0))

    # 1. WARMUP: MUST MATCH THE BENCHMARK EXACTLY (60 steps)
    # If we warmup with 1 step, JAX will RE-COMPILE when we ask for 60.
    print("... Compiling Kernel (Warmup with 60 steps) ...")
    synchronize(topology.multi_step_successor(start_v, 60))

    # 2. MEASURE
    t0 = time.perf_counter()
    result_subset = topology.multi_step_successor(start_v, 60)
    synchronize(result_subset)
    dt = (time.perf_counter() - t0) * 1000
    # Verify
    count = getattr(result_subset, 'num_states', 0)
    print(f"🚀 Time: {dt:.4f} ms")
    print(f"-> Reachable States: {count}")

    # ==========================================
    # B. FIELDS: Clean User Experience
    # ==========================================
    print("\n[B] Field Algebra (Add + Normalize)")

    alg = RealFieldAlgebra()
    f1 = FieldMapper.define_constant_field(space, alg, 1.0)
    f2 = FieldMapper.define_constant_field(space, alg, 2.0)

    # Warmup
    synchronize(FieldSpaceComposition.apply(f1, f2, RealAddition(), alg))

    # MEASURE
    t0 = time.perf_counter()

    sum_field = FieldSpaceComposition.apply(f1, f2, RealAddition(), alg)

    mass = float(jnp.sum(sum_field.explicit_buffer))
    norm_field = FieldSpaceTransform.apply(sum_field, ScaleTransform(1.0 / mass), alg)

    synchronize(norm_field)
    dt = (time.perf_counter() - t0) * 1000

    print(f"🚀 Time: {dt:.4f} ms")

    val = float(norm_field.get_fields_at(0).value.ravel()[0])
    print(f"-> Final Value: {val:.8f}")
    assert val > 0.0
    print("✅ Verified")


if __name__ == "__main__":
    run_benchmark()