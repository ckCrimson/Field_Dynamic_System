import pytest
import jax
import jax.numpy as jnp
import numpy as np

# --- IMPORTS FROM REAL SOURCE (NO REDEFINITION) ---
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteStateSpace
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.core.field.transform import FieldTransform
from src.field_dynamic_system.neighbor.discrete import GraphTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import UnbiasedKernel


# --- 1. THE DECOUPLED NORMALIZER ---
# This is the "User Logic" we inject into the Generator
class EdgeNormalizer(FieldTransform):
    def __init__(self, scale_vector):
        # We ensure the vector is shaped (Num_Edges, 1) to match the Generator's edge flow
        self.scale = jnp.array(scale_vector).reshape(-1, 1)

    def __call__(self, edge_vals):
        # This aligns with Step 3 (Intrinsic) in your pipeline
        return edge_vals * self.scale


# --- 2. THE TEST ---
def test_real_generator_mass_conservation():
    print("\n==================================================")
    print("🛡️ REAL SYSTEM TEST: Generator + Intrinsic Normalizer")
    print("==================================================")

    # 1. TOPOLOGY: Hub (0) <-> 10 Leaves (1..10)
    edges = []
    for i in range(1, 11):
        edges.append((0, i))
        edges.append((i, 0))

    states = list(range(11))
    space = DiscreteStateSpace(states)
    topo = GraphTopology(space, edges=edges)

    # 2. FIELD
    algebra = RealFieldAlgebra()
    mapper = DiscreteFieldMapper(space, algebra)
    mapper.set_value_at(0, 100.0)

    # 3. SETUP TRANSFORM DATA (Crucial Step)
    # We query the topology to build the correct normalization vector
    raw_kernel = UnbiasedKernel(prob=1.0)
    matrix = topo.get_adjacency_matrix()

    # Calculate Out-Degree
    src_indices = matrix.indices[:, 1]
    num_nodes = matrix.shape[0]
    weights = matrix.data

    # Sum weights per source node
    row_sums = jax.ops.segment_sum(weights, src_indices, num_segments=num_nodes)

    # Map back to edges to create the 1/Degree vector
    edge_degrees = row_sums[src_indices]
    norm_vector = 1.0 / jnp.where(edge_degrees == 0, 1.0, edge_degrees)

    # Instantiate the Transform
    normalizer = EdgeNormalizer(norm_vector)

    # 4. INSTANTIATE REAL GENERATOR
    # We use the existing class from src
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=topo,
        kernel=raw_kernel,
        step_composer=AdditionComposition(),
        chain_composer=None,
        intrinsic_transform=normalizer,  # <--- INJECTED HERE
        extrinsic_transform=None
    )

    # 5. RUN
    print("-> Running 100 Steps...")
    result_mapper = gen.generate_multi_step(mapper, steps=100)

    # 6. VERIFY
    final_mass = float(jnp.sum(result_mapper.raw_buffer))
    final_vec = np.array(result_mapper.raw_buffer.flatten())

    print(f"-> Final Mass: {final_mass:.4f}")
    print(f"-> Hub Value:  {final_vec[0]:.4f}")

    # Strict Mass Check
    assert abs(final_mass - 100.0) < 1e-4, f"Mass Leak! Got {final_mass}"
    assert final_vec[0] > 99.0, "Mass did not return to Hub."

    print("✅ SUCCESS: The existing Generator architecture works perfectly.")


if __name__ == "__main__":
    test_real_generator_mass_conservation()