import pytest
import random
import jax.numpy as jnp
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.neighbor.discrete import GraphTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import UnbiasedKernel
from src.field_dynamic_system.core.field.compositions import AdditionComposition


def test_large_scale_saturation():
    print("\n==================================================")
    print("🏙️  LARGE SCALE TEST: 100 Nodes, Connectivity Saturation")
    print("==================================================")

    # 1. SETUP: 100 Nodes, 5 Random Connections each
    NUM_NODES = 100
    CONNECTIONS = 6
    node_names = [f"R_{i}" for i in range(NUM_NODES)]

    # Deterministic Randomness for consistent tests
    rng = random.Random(42)
    edges = []
    for source in node_names:
        targets = rng.sample(node_names, CONNECTIONS)
        for t in targets:
            if t != source:
                edges.append((source, t))

    print(f"-> Topology: {NUM_NODES} Nodes, {len(edges)} Edges.")

    # 2. INITIALIZE SYSTEM
    state_space = AbstractDiscreteStateSpace(node_names)
    topology = GraphTopology(state_space, edges=edges)

    # Broadcast Physics: Signal adds up and spreads everywhere
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=topology,
        kernel=UnbiasedKernel(prob=1.0),
        step_composer=AdditionComposition()
    )

    mapper = DiscreteFieldMapper(state_space, VectorFieldAlgebra(dim=1))

    # 3. INJECT SIGNAL (At Node 0)
    start_node = "R_0"
    mapper.set_value_at(start_node, 100.0)
    print(f"-> Injection at {start_node}. Starting Simulation...")

    # 4. RUN LOOP (10 Steps)
    current_mapper = mapper

    print(f"\n{'STEP':<5} | {'ACTIVE NODES':<15} | {'COVERAGE':<10} | {'TOTAL SIGNAL':<15}")
    print("-" * 55)

    fully_connected = False

    for t in range(1, 11):
        # Evolve
        current_mapper = gen.generate_multi_step(current_mapper, steps=1)

        # ANALYSIS
        # Get raw buffer to count how many nodes have signal > 0
        raw_data = current_mapper.raw_buffer  # Shape (100, 1)
        active_count = jnp.sum(raw_data > 0.001)  # Count non-zeros
        total_signal = jnp.sum(raw_data)

        coverage = (active_count / NUM_NODES) * 100

        print(f"{t:<5} | {active_count:<15} | {coverage:5.1f}%     | {total_signal:.2e}")

        # CHECK FOR SATURATION
        if active_count == NUM_NODES:
            print(f"\n✅ FULL CONNECTION ACHIEVED at Step {t}!")
            fully_connected = True
            break

    assert fully_connected, "Network failed to fully connect in 10 steps!"


if __name__ == "__main__":
    test_large_scale_saturation()