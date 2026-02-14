import pytest
import time
import jax.numpy as jnp
import numpy as np
import jax

from src.field_dynamic_system.core import FieldTransform
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import VectorGridTopology, WeightedVectorGridTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import AdditionComposition


class DummyKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper):
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)


class SimpleNormalizer(FieldTransform):
    def __call__(self, field): return field


def test_senior_engineer_benchmark():
    print("\n==================================================")
    print("🏆 DUAL-PATH BENCHMARK: 4D Hypercube Random Walker")
    print("==================================================")

    DIMENSIONS = 4
    STEPS = 40

    T0_TOTAL = time.time()

    # 1. SETUP & DISCOVER
    seed_tuple = tuple([0] * DIMENSIONS)
    temp_space = VectorStateSpace([VectorState(seed_tuple)], dim=DIMENSIONS)
    deltas = np.eye(DIMENSIONS, dtype=int).tolist()

    print(f"-> Phase 1: Mapping 4D Space (Depth={STEPS})...")
    discovery_topo = VectorGridTopology(temp_space, deltas)
    discovery_topo.expand_frontier([seed_tuple], depth=STEPS)
    raw_states, _ = discovery_topo.export_discovery()
    count = len(raw_states)
    print(f"   States Mapped:  {count:,}")

    # 2. BUILD OOP OBJECTS
    print("-> Phase 2: Building OOP System...")
    raw_matrix = np.array(raw_states, dtype=np.float32)
    real_space = VectorStateSpace.from_raw_data(
        raw_matrix, wrapper=lambda x: VectorState(tuple(x.tolist())), dim=DIMENSIONS
    )
    algebra = VectorFieldAlgebra(dim=1)
    mapper = DiscreteFieldMapper(real_space, algebra)
    mapper.sync_size(count)
    mapper.apply_vector(jnp.zeros((count, 1)).at[0].set(1.0))

    real_topo = WeightedVectorGridTopology(real_space, deltas, weight=1.0 / DIMENSIONS)
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=real_topo, kernel=DummyKernel(),
        intrinsic_transform=SimpleNormalizer(), extrinsic_transform=SimpleNormalizer(),
        step_composer=AdditionComposition(), chain_composer=None, global_composer=None
    )
    _ = gen.topology.adjacency_matrix

    # 3. RUN PHYSICS (OOP/MAPPER PATH)
    print("\n-> Phase 3: Running Simulation (OOP Mapper Path)...")
    # Warmup
    gen.generate_multi_step(mapper, steps=1).raw_buffer.block_until_ready()

    t0_run = time.time()
    res_oop = gen.generate_multi_step(mapper, steps=STEPS)
    final_vec_oop = res_oop.raw_buffer.block_until_ready()
    time_oop = time.time() - t0_run
    print(f"   OOP Run Time: {time_oop:.4f}s")

    # 4. RUN PHYSICS (RAW PATH)
    print("\n-> Phase 4: Running Simulation (Stateless Raw Path)...")

    # Extract Raw Components just like the System will do
    raw_initial_field = mapper.raw_buffer
    adj_mat = gen.topology.adjacency_matrix
    raw_topo_tuple = (
        adj_mat.indices[:, 0],  # tgt_indices
        adj_mat.indices[:, 1],  # src_indices
        adj_mat.data,  # weights
        count  # num_nodes
    )

    # Warmup
    gen.generate_raw_multi_step(raw_initial_field, raw_topo_tuple, steps=1).block_until_ready()

    t0_raw = time.time()
    final_vec_raw = gen.generate_raw_multi_step(raw_initial_field, raw_topo_tuple, steps=STEPS).block_until_ready()
    time_raw = time.time() - t0_raw
    print(f"   Raw Run Time: {time_raw:.4f}s")

    # 5. VERIFY
    TOTAL_TIME = time.time() - T0_TOTAL
    final_prob = float(jnp.sum(final_vec_raw))

    print("\n==================================================")
    print("📊 FINAL METRICS")
    print("==================================================")
    print(f"Total Wall Time:   {TOTAL_TIME:.4f}s")
    print(f"States Processed:  {count:,}")
    print(f"Mass Conservation: {final_prob:.4f}")

    # MATH CONSISTENCY CHECK
    is_identical = jnp.allclose(final_vec_oop, final_vec_raw)
    print(f"Output Consistency: {'✅ 100% Identical' if is_identical else '❌ Mismatch'}")

    assert is_identical, "The Raw path produced different math than the OOP path!"
    assert 0.99 < final_prob < 1.01, "Mass Leak Detected"


if __name__ == "__main__":
    test_senior_engineer_benchmark()