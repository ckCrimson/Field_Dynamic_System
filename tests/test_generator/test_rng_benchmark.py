import pytest
import time
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import VectorGridTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator, \
    GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import AdditionComposition


# --- FIX 1: WEIGHTED TOPOLOGY (Mass Conservation) ---
class WeightedVectorGridTopology(VectorGridTopology):
    def __init__(self, space, deltas, weight=1.0):
        super().__init__(space, deltas)
        self.weight = weight

    # We override the builder to inject weights (e.g. 0.166) instead of 1.0
    def _build_fast_vector_matrix(self) -> sparse.BCOO:
        # Re-using the Ferrari Engine logic but with weights
        raw_coords = np.array(self.discrete_space.get_matrix(), dtype=np.int64)
        N, D = raw_coords.shape

        # Encoding & Sorting
        bounds_max = int(np.max(raw_coords)) + 2
        strides = np.array([bounds_max ** i for i in range(D)], dtype=np.int64)
        encoded_states = raw_coords.dot(strides)
        sort_perm = np.argsort(encoded_states)
        sorted_encoded = encoded_states[sort_perm]

        sources, targets = [], []
        deltas = np.array(self.deltas, dtype=np.int64)

        for delta in deltas:
            delta_val = np.dot(delta, strides)
            potential_neighbors = sorted_encoded + delta_val
            positions = np.searchsorted(sorted_encoded, potential_neighbors)
            positions = np.clip(positions, 0, N - 1)
            found_mask = (sorted_encoded[positions] == potential_neighbors)

            valid_src = sort_perm[np.where(found_mask)[0]]
            valid_tgt = sort_perm[positions[found_mask]]
            sources.append(valid_src)
            targets.append(valid_tgt)

        if not sources: return sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))

        all_sources = np.concatenate(sources)
        all_targets = np.concatenate(targets)
        indices = jnp.column_stack((jnp.array(all_targets), jnp.array(all_sources)))

        # CRITICAL FIX: Use self.weight instead of 1.0
        values = jnp.full((len(all_sources),), self.weight, dtype=jnp.float32)

        return sparse.BCOO((values, indices), shape=(N, N))


class SimpleNormalizer:
    def __call__(self, field): return field


class DummyKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper):
        # The weights are now in the matrix, so the kernel is just a pass-through
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)


def test_honest_dynamic_benchmark():
    print("\n==================================================")
    print("🌍 REAL HONEST BENCHMARK (End-to-End)")
    print("==================================================")

    # START THE CLOCK (Total Wall Time)
    T0_TOTAL = time.time()

    DIMENSIONS = 15
    STEPS = 5

    # 1. INITIALIZATION
    seed_tuple = tuple([0] * DIMENSIONS)
    seed_state = VectorState(seed_tuple)
    temp_space = VectorStateSpace([seed_state], dim=DIMENSIONS)
    deltas = np.eye(DIMENSIONS, dtype=int).tolist()

    # Use standard topo for discovery (finding the map)
    discovery_topo = VectorGridTopology(temp_space, deltas)

    # 2. DISCOVERY PHASE (CPU Work)
    print(f"-> Phase 1: Discovering World (Depth={STEPS})...")
    t0_disc = time.time()
    discovery_topo.expand_frontier([seed_tuple], depth=STEPS)
    raw_states, _ = discovery_topo.export_discovery()
    time_disc = time.time() - t0_disc

    print(f"   States Found:   {len(raw_states):,}")
    print(f"   Discovery Time: {time_disc:.4f}s")

    # 3. PHYSICS SETUP (Matrix Build)
    print("-> Phase 2: Building Weighted Physics Engine...")
    t0_build = time.time()

    raw_matrix = np.array(raw_states, dtype=np.float32)
    real_space = VectorStateSpace.from_raw_data(
        raw_matrix, wrapper=lambda x: VectorState(tuple(x.tolist())), dim=DIMENSIONS
    )

    algebra = VectorFieldAlgebra(dim=1)
    mapper = DiscreteFieldMapper(real_space, algebra)
    mapper.sync_size(len(raw_states))
    mapper.apply_vector(jnp.zeros((len(raw_states), 1)).at[0].set(1.0))

    # USE WEIGHTED TOPOLOGY (1/6 probability per edge)
    real_topo = WeightedVectorGridTopology(real_space, deltas, weight=1.0 / DIMENSIONS)

    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=real_topo,
        kernel=DummyKernel(),  # Weights handled by topology
        intrinsic_transform=SimpleNormalizer(),
        extrinsic_transform=SimpleNormalizer(),
        step_composer=AdditionComposition(),
        chain_composer=None,
        global_composer=None
    )

    # Force Build
    _ = gen.topology.adjacency_matrix
    time_build = time.time() - t0_build
    print(f"   Matrix Build:   {time_build:.4f}s")

    # 4. SIMULATION RUN (JAX GPU)
    print("-> Phase 3: Running Simulation...")
    # Warmup (compiled time)
    gen.generate_multi_step(mapper, steps=1).raw_buffer.block_until_ready()

    t0_run = time.time()
    res = gen.generate_multi_step(mapper, steps=STEPS)
    final_vec = res.raw_buffer.block_until_ready()
    time_run = time.time() - t0_run

    print(f"   Physics Time:   {time_run:.4f}s")

    # --- FINAL METRICS ---
    TOTAL_TIME = time.time() - T0_TOTAL

    print("\n==================================================")
    print("📊 FINAL RESULTS")
    print("==================================================")
    print(f"Total Wall Time:  {TOTAL_TIME:.4f}s")
    print(f"True Throughput:  {STEPS / TOTAL_TIME:.2f} Steps/Sec (End-to-End)")
    print(f"Physics Speed:    {STEPS / time_run:.0f} Steps/Sec (Burst)")

    # VERIFY PHYSICS
    final_prob = float(jnp.sum(final_vec))
    print(f"Total Probability: {final_prob:.4f}")

    # Should be exactly 1.0 (Mass Conserved)
    # Allow tiny float error
    assert 0.99 < final_prob < 1.01, f"Mass NOT conserved: {final_prob}"
    print("✅ Physics Valid: Mass Conserved")


if __name__ == "__main__":
    test_honest_dynamic_benchmark()