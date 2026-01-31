import pytest
import jax.numpy as jnp
import jax.ops
from src.field_dynamic_system.core.field.compositions import FieldComposition, AdditionComposition
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.neighbor.discrete import GraphTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import UnbiasedKernel


# --- 1. DEFINE CUSTOM PHYSICS (MAX COMPOSITION) ---
class MaxComposition(FieldComposition):
    """
    Takes the MAXIMUM value of all incoming signals.
    Used for 'Flood Fill' or 'Strongest Signal' physics.
    """

    def compose(self, field_a, field_b):
        # Element-wise Max (Standard)
        return jnp.maximum(field_a, field_b)

    def compose_reduction(self, weighted_vals, tgt_indices, num_nodes):
        """
        The critical hook for the Generator.
        Reduces many incoming edge values into a single node value using MAX.
        """
        # Initialize with -inf so max works correctly
        init_val = jnp.full((num_nodes, 1), -jnp.inf, dtype=weighted_vals.dtype)

        # JAX Segment Max: Efficiently computes max per group (target index)
        # Note: We assume weighted_vals is shape (Edges, 1)
        # tgt_indices is shape (Edges,)
        return jax.ops.segment_max(weighted_vals, tgt_indices, num_segments=num_nodes, indices_are_sorted=False)


def test_non_linear_physics():
    print("\n==================================================")
    print("🌊 GENERIC PHYSICS TEST: Max Composition (Flood)")
    print("==================================================")

    # 1. TOPOLOGY: 2 Sources -> 1 Target
    nodes = ["Target", "High_Source", "Low_Source"]
    # Edges: High->Target, Low->Target
    edges = [
        ("High_Source", "Target"),
        ("Low_Source", "Target")
    ]

    space = DiscreteStateSpace(nodes)
    topo = GraphTopology(space, edges)

    # 2. INITIALIZE FIELD
    # High = 10.0, Low = 2.0, Target = 0.0
    mapper = DiscreteFieldMapper(space, VectorFieldAlgebra(dim=1))
    mapper.set_value_at("High_Source", 10.0)
    mapper.set_value_at("Low_Source", 2.0)

    print(f"-> Initial State:\n   High=10.0\n   Low=2.0\n   Target=0.0")

    # 3. BUILD GENERATOR WITH MAX COMPOSITION
    # We use UnbiasedKernel (Weight=1.0) so values are passed unchanged
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=topo,
        kernel=UnbiasedKernel(prob=1.0),
        step_composer=MaxComposition(),  # <--- THE KEY CHANGE
        chain_composer=None  # Memoryless (New replaces Old)
    )

    # 4. RUN 1 STEP
    result = gen.generate_multi_step(mapper, steps=1)

    # 5. INSPECT RESULTS
    res_dict = result.export_to_dict()
    val_target = float(res_dict["Target"][0])

    print(f"-> Result at Target: {val_target}")

    # 6. VERIFICATION
    # If logic was ADDITION: 10 + 2 = 12.0
    # If logic is MAX: max(10, 2) = 10.0

    if abs(val_target - 12.0) < 0.001:
        print("❌ FAILED: System used ADDITION logic instead of MAX.")
        pytest.fail("Generator ignored custom step_composer!")

    elif abs(val_target - 10.0) < 0.001:
        print("✅ PASSED: System correctly used MAX logic.")
    else:
        print(f"❌ FAILED: Unexpected value {val_target}")
        pytest.fail("Calculation error.")


if __name__ == "__main__":
    test_non_linear_physics()