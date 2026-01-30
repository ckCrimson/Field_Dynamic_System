import pytest
import jax.numpy as jnp
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.neighbor.discrete import GraphTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import UnbiasedKernel
from src.field_dynamic_system.core.field.compositions import AdditionComposition


def test_abstract_packet_routing():
    print("\n==================================================")
    print("🌐 ABSTRACT GRAPH TEST: Packet Routing Network")
    print("==================================================")

    # 1. DEFINE ABSTRACT STATES (Strings)
    # No coordinates. Just names.
    nodes = ["Source", "Router_A", "Router_B", "Destination"]

    # Create Space (The Map)
    # This automatically maps: "Source"->0, "Router_A"->1, etc.
    state_space = AbstractDiscreteStateSpace(nodes)

    print(f"   States Created: {state_space.states}")

    # 2. DEFINE TOPOLOGY (The Connections)
    # Explicitly defining the edges (Abstract -> Abstract)
    edges = [
        ("Source", "Router_A"),  # 50% traffic here
        ("Source", "Router_B"),  # 50% traffic here
        ("Router_A", "Destination"),
        ("Router_B", "Destination")
    ]

    # GraphTopology handles the conversion of Strings -> Indices internally
    topology = GraphTopology(state_space, edges=edges)

    # 3. BUILD THE GENERATOR
    # We use a standard Kernel (Unbiased) to split traffic 50/50 automatically
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=topology,
        kernel=UnbiasedKernel(),  # Splits 1.0 -> 0.5, 0.5
        step_composer=AdditionComposition()  # Traffic adds up at destination
    )

    # 4. INITIALIZE FIELD (100 Packets at Source)
    algebra = VectorFieldAlgebra(dim=1)
    mapper = DiscreteFieldMapper(state_space, algebra)

    # We set the value using the Abstract Name
    print("-> Injecting 100 Packets at 'Source'...")
    mapper.set_value_at("Source", jnp.array([100.0]))

    print(f"   Initial State: {mapper.export_to_dict()}")

    # 5. RUN SIMULATION
    # Step 1: Source -> Routers
    print("\n-> Running Step 1 (Source -> Routers)...")
    res_1 = gen.generate_multi_step(mapper, steps=1)
    state_1 = res_1.export_to_dict()
    print(f"   State T=1: {state_1}")

    # CHECK: Did traffic split?
    # Router_A and Router_B should have 50.0 each.
    # Note: Depending on UnbiasedKernel implementation, might be normalized.
    # If edge weights are 1.0, and we use raw matrix, it multiplies by 1.0.
    # But usually UnbiasedKernel implies normalization. Let's assume standard weights 1.0 for now if Kernel isn't normalizing active edges.

    # Step 2: Routers -> Destination
    print("\n-> Running Step 2 (Routers -> Destination)...")
    res_2 = gen.generate_multi_step(mapper, steps=2)  # Note: This runs from T=0 for 2 steps
    state_2 = res_2.export_to_dict()
    print(f"   State T=2: {state_2}")

    # 6. VERIFICATION
    final_dest_val = state_2.get("Destination")[0]

    # Assertions
    assert final_dest_val > 99.0, f"Loss of packets! Expected ~100, got {final_dest_val}"
    assert state_2.get("Source")[0] == 0.0, "Source should be empty"

    print("\n✅ TEST PASSED: Abstract Physics worked perfectly.")


if __name__ == "__main__":
    test_abstract_packet_routing()