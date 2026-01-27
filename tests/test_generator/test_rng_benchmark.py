import pytest
import time
import jax.numpy as jnp
import numpy as np
import itertools

# --- IMPORTS ---
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import AdditionComposition


# ==============================================================================
# 1. TEST COMPONENTS
# ==============================================================================

class UnbiasedKernel(AbstractTransitionKernel):
    def __init__(self, prob): self.p = prob

    def compute_raw_batch(self, edge_indices, context_mapper):
        return jnp.full((edge_indices.shape[0], 1), self.p, dtype=jnp.float32)


class RNGTopology(DiscreteTopology):
    def __init__(self, space, dims):
        super().__init__(space)
        self.deltas = np.eye(dims, dtype=int)

    def compute_neighbors(self, state_obj):
        # Assumes state_obj.values is a tuple
        base = np.array(state_obj.values)
        neighbors = []
        for d in self.deltas:
            new_vals = tuple((base + d).tolist())
            neighbors.append(VectorState(new_vals))
        return neighbors


# ==============================================================================
# 2. THE INSTANT BENCHMARK
# ==============================================================================

def test_rng_benchmark_safe():
    print("\n==================================================")
    print("🚀 BENCHMARK: Validity Check (Safe Parameters)")
    print("==================================================")

    D = 3

    # --- TUNED PARAMETERS (To prevent O(N^2) hang) ---
    BENCH_DEPTH = 15  # ~800 States (Small enough for linear search)
    BENCH_STEPS = 10  # 10 Steps < 15 Depth (Token stays inside)

    # 1. BUILD SPACE
    print(f"-> Generating States (Depth={BENCH_DEPTH})...")
    grid = [VectorState(c) for c in itertools.product(range(BENCH_DEPTH + 1), repeat=D)
            if sum(c) <= BENCH_DEPTH]
    space = VectorStateSpace(grid, dim=D)
    print(f"   States: {len(grid)}")

    # 2. INITIALIZE FIELD
    algebra = VectorFieldAlgebra(dim=1)
    mapper = DiscreteFieldMapper(space, algebra)
    mapper.sync_size(len(grid))

    # Set Start
    start_idx = space.get_index_of(VectorState((0, 0, 0)))
    mapper.apply_vector(jnp.zeros((len(grid), 1)).at[start_idx].set(1.0))

    # 3. GENERATOR (Matrix Build)
    print("-> Building Adjacency Matrix...")
    t0_init = time.time()

    gen = DiscreteFieldGenerator(
        topology=RNGTopology(space, D),
        kernel=UnbiasedKernel(prob=1.0 / D),
        chain_composer=None  # Default Markov Replacement
    )
    print(f"   Init Time: {time.time() - t0_init:.4f}s")

    # 4. RUN (JAX)
    print("-> JIT & Run...")
    t0_run = time.time()
    res = gen.generate_multi_step(mapper, steps=BENCH_STEPS)
    final_vec = res.raw_buffer.block_until_ready()
    dt = time.time() - t0_run

    # 5. RESULTS
    final_prob = float(jnp.sum(final_vec))
    sps = BENCH_STEPS / dt

    print(f"\n   Time: {dt:.4f}s")
    print(f"   SPS:  {sps:,.0f}")
    print(f"   Prob: {final_prob:.4f}")

    assert abs(final_prob - 1.0) < 1e-4, f"Leaked Prob: {final_prob}"
    print("\n✅ SUCCESS: Generator Works.")


if __name__ == "__main__":
    test_rng_benchmark_safe()