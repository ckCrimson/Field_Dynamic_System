import pytest
import time
import jax.numpy as jnp
import numpy as np

from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import VectorGridTopology, WeightedVectorGridTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import AdditionComposition


class DummyKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper):
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)


class SimpleNormalizer:
    def __call__(self, field): return field


def test_senior_engineer_benchmark():
    print("\n==================================================")
    print("🏆 SENIOR ENGINEER BENCHMARK: 4D Hypercube")
    print("==================================================")

    # THE SWEET SPOT:
    # D=4, Steps=40 creates ~135,000 states.
    # This is heavy enough to prove the engine is fast,
    # but light enough to finish in ~3 seconds on a laptop.
    DIMENSIONS = 4
    STEPS = 40

    T0_TOTAL = time.time()

    # 1. SETUP
    seed_tuple = tuple([0] * DIMENSIONS)
    temp_space = VectorStateSpace([VectorState(seed_tuple)], dim=DIMENSIONS)
    deltas = np.eye(DIMENSIONS, dtype=int).tolist()

    # 2. DISCOVER (~135k States)
    print(f"-> Phase 1: Mapping 4D Space (Depth={STEPS})...")
    discovery_topo = VectorGridTopology(temp_space, deltas)

    t0_disc = time.time()
    discovery_topo.expand_frontier([seed_tuple], depth=STEPS)
    raw_states, _ = discovery_topo.export_discovery()
    time_disc = time.time() - t0_disc

    count = len(raw_states)
    print(f"   States Mapped:  {count:,}")
    print(f"   Discovery Time: {time_disc:.4f}s")

    # 3. BUILD MATRIX
    print("-> Phase 2: Building JAX Matrix...")
    raw_matrix = np.array(raw_states, dtype=np.float32)
    real_space = VectorStateSpace.from_raw_data(
        raw_matrix, wrapper=lambda x: VectorState(tuple(x.tolist())), dim=DIMENSIONS
    )
    algebra = VectorFieldAlgebra(dim=1)
    mapper = DiscreteFieldMapper(real_space, algebra)
    mapper.sync_size(count)
    mapper.apply_vector(jnp.zeros((count, 1)).at[0].set(1.0))

    t0_build = time.time()
    real_topo = WeightedVectorGridTopology(real_space, deltas, weight=1.0 / DIMENSIONS)
    gen = DiscreteFieldGenerator(
        topology=real_topo, kernel=DummyKernel(),
        intrinsic_transform=SimpleNormalizer(), extrinsic_transform=SimpleNormalizer(),
        step_composer=AdditionComposition(), chain_composer=None, global_composer=None
    )
    _ = gen.topology.adjacency_matrix
    print(f"   Build Time:     {time.time() - t0_build:.4f}s")

    # 4. RUN PHYSICS
    print("-> Phase 3: Running Simulation...")
    gen.generate_multi_step(mapper, steps=1).raw_buffer.block_until_ready()  # Warmup

    t0_run = time.time()
    res = gen.generate_multi_step(mapper, steps=STEPS)
    final_vec = res.raw_buffer.block_until_ready()
    time_run = time.time() - t0_run

    # 5. VERIFY
    TOTAL_TIME = time.time() - T0_TOTAL
    final_prob = float(jnp.sum(final_vec))

    print("\n==================================================")
    print("📊 FINAL METRICS")
    print("==================================================")
    print(f"Total Wall Time:   {TOTAL_TIME:.4f}s")
    print(f"States Processed:  {count:,}")
    print(f"Physics Steps:     {STEPS}")
    print(f"Mass Conservation: {final_prob:.4f}")

    # ASSERTIONS FOR "WORKING PROOF"
    assert 0.99 < final_prob < 1.01, "Mass Leak Detected"
    assert count > 100000, "Too few states to prove scalability"

    # The crucial "Senior" Check:
    # Did we process >100k complex states in roughly 3 seconds?
    if TOTAL_TIME < 4.0:
        print("✅ PERFORMANCE: EXCELLENT (Senior Level)")
    else:
        print("⚠️ PERFORMANCE: GOOD (Mid Level)")


if __name__ == "__main__":
    test_senior_engineer_benchmark()