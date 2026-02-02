import pytest
import random
import jax
import jax.numpy as jnp
import numpy as np

# --- IMPORTS ---
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.neighbor.discrete import GraphTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import UnbiasedKernel
from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.core.field.transform import FieldTransform


# --- 1. THE NORMALIZER ---
class EdgeNormalizer(FieldTransform):
    def __init__(self, scale_vector):
        self.scale = jnp.array(scale_vector).reshape(-1, 1)

    def __call__(self, edge_vals):
        return edge_vals * self.scale


def test_large_scale_saturation_normalized():
    print("\n==================================================")
    print("🏙️  LARGE SCALE TEST: 100 Nodes (Normalized)")
    print("==================================================")

    # 1. SETUP: 100 Nodes, 6 Random Connections each
    NUM_NODES = 100
    CONNECTIONS = 5
    node_names = [f"R_{i}" for i in range(NUM_NODES)]

    # Deterministic Randomness
    rng = random.Random(42)
    edges = []
    for source in node_names:
        targets = rng.sample(node_names, CONNECTIONS)
        for t in targets:
            if t != source:
                edges.append((source, t))

    print(f"-> Topology: {NUM_NODES} Nodes, {len(edges)} Edges.")

    # 2. INITIALIZE SYSTEM
    state_space = DiscreteStateSpace(node_names)
    topology = GraphTopology(state_space, edges=edges)

    # 3. SETUP NORMALIZER (The Fix)
    # Get raw structure to calculate degrees
    raw_kernel = UnbiasedKernel(prob=1.0)
    matrix = topology.get_adjacency_matrix()

    # Calculate Out-Degree
    src_indices = matrix.indices[:, 1]
    num_nodes = matrix.shape[0]
    weights = matrix.data

    # Sum weights per source node (Total outgoing weight)
    row_sums = jax.ops.segment_sum(weights, src_indices, num_segments=num_nodes)

    # Map back to edges to get "My Source's Degree"
    edge_degrees = row_sums[src_indices]

    # Normalization Factor = 1.0 / Degree
    norm_vector = 1.0 / jnp.where(edge_degrees == 0, 1.0, edge_degrees)

    normalizer = EdgeNormalizer(norm_vector)

    # 4. SETUP GENERATOR
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=topology,
        kernel=raw_kernel,
        step_composer=AdditionComposition(),
        chain_composer=None,  # Replacement Logic (Standard Markov)
        intrinsic_transform=normalizer  # <--- INJECTED
    )

    mapper = DiscreteFieldMapper(state_space, VectorFieldAlgebra(dim=1))

    # 5. INJECT SIGNAL
    start_node = "R_0"
    mapper.set_value_at(start_node, 100.0)
    print(f"-> Injection at {start_node} (100.0). Starting Simulation...")

    # 6. RUN LOOP
    current_mapper = mapper
    fully_connected = False

    print(f"\n{'STEP':<5} | {'ACTIVE NODES':<15} | {'COVERAGE':<10} | {'TOTAL SIGNAL':<15}")
    print("-" * 55)

    for t in range(1, 11):
        current_mapper = gen.generate_multi_step(current_mapper, steps=1)

        # ANALYSIS
        raw_data = current_mapper.raw_buffer
        active_count = jnp.sum(raw_data > 0.001)
        total_signal = float(jnp.sum(raw_data))
        coverage = (active_count / NUM_NODES) * 100

        print(f"{t:<5} | {active_count:<15} | {coverage:5.1f}%     | {total_signal:.4f}")

        # MASS CONSERVATION CHECK
        if abs(total_signal - 100.0) > 1e-4:
            print(f"❌ Mass Explosion Detected! {total_signal}")
            pytest.fail("Normalization Failed")

        # SATURATION CHECK
        if active_count == NUM_NODES:
            print(f"\n✅ FULL CONNECTION ACHIEVED at Step {t}!")
            fully_connected = True
            break

    assert fully_connected, "Network failed to fully connect in 10 steps!"
    print("✅ SUCCESS: Mass Conserved & Network Saturated.")


if __name__ == "__main__":
    test_large_scale_saturation_normalized()