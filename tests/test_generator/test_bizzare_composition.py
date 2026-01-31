import jnp
import pytest
import jax.numpy as jnp
import jax.ops

from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteStateSpace
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.compositions import FieldComposition, AdditionComposition, \
    ClosedFieldComposition
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import ElementwiseKernel
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from tests.test_core_field.proof_of_concept_number_line import RealFieldAlgebra


# =========================================================================
# 1. THE BIZARRE LOGIC
# =========================================================================
class BizarreComposition(ClosedFieldComposition):
    """
    Rules:
    1. If (a=Null AND b=Null) -> 3.0
    2. If (a=Null OR  b=Null) -> (The_Present_Value + 2.0)
    3. Else (Both Present)    -> a + b
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        zeros = 0.0

        # Conditions
        a_is_null = (a == zeros)
        b_is_null = (b == zeros)
        both_null = a_is_null & b_is_null
        one_null = (a_is_null | b_is_null) & (~both_null)  # XOR-like

        # Rule 3: Both Null -> 3.0
        val_both_null = 3.0

        # Rule 2: One Null -> Present + 2.0
        # If a is null, use b. If b is null, use a.
        present_val = jnp.where(a_is_null, b, a)
        val_one_null = present_val + 2.0

        # Rule 4: Else -> a + b
        val_else = a + b

        # Combine (Priority: BothNull > OneNull > Else)
        # We nest jnp.where to handle the branching
        result = jnp.where(both_null, val_both_null,
                           jnp.where(one_null, val_one_null, val_else))

        return result

    def get_identity(self) -> float:
        return 0.0  # Irrelevant here as we handle nulls internally

    # Default reduction is Sum, which is what we want for this test


# =========================================================================
# 2. SETUP
# =========================================================================

# Space: -3 to 3 (Small enough to debug manually)
raw_states = list(range(-3, 4))  # [-3, -2, -1, 0, 1, 2, 3]
space = DiscreteStateSpace(raw_states)
algebra = RealFieldAlgebra()

# Initial Field: 1s at -1 and 1. Zeros elsewhere.
# Indices: -3(0), -2(1), -1(2), 0(3), 1(4), 2(5), 3(6)
# Target indices for setup: idx 2 (-1) and idx 4 (1)
initial_buffer = jnp.zeros((7, 1))
initial_buffer = initial_buffer.at[2].set(1.0)
initial_buffer = initial_buffer.at[4].set(1.0)

# Create Mapper manually with this buffer
field_mapper = DiscreteFieldMapper(space, algebra, explicit_buffer=initial_buffer)


# Topology: Neighbors [s-1, s+1]
class OneDimWalkerTopology(DiscreteTopology):
    def compute_neighbors(self, state_0):
        return [state_0 - 1, state_0 + 1]


# Kernel: Unity (No weight change)
class UnityKernel(ElementwiseKernel):
    def compute_transition_value(self, state_in, state_out):
        return 1.0


# =========================================================================
# 3. TEST EXECUTION
# =========================================================================
def test_bizarre_composition_manual():
    print("\n==================================================")
    print("🤪 BIZARRE COMPOSITION TEST")
    print("==================================================")

    topology = OneDimWalkerTopology(space)

    # We use NO normalizers (Identity) to keep integers raw.
    generator = GenericMarkovianDiscreteFieldGenerator(
        topology=topology,
        kernel=UnityKernel(),
        step_composer=AdditionComposition(),  # Sum the results
        chain_composer=BizarreComposition(),  # Apply Bizarre Rules
        global_composer=None
    )

    # Run 1 Step
    result = generator.generate_multi_step(field_mapper, steps=1)

    # --- HELPER: Read Result ---
    def get_res(val):
        idx = space.get_index_of(val)
        return float(result.raw_buffer[idx][0])

    # --- MANUAL VERIFICATION ---

    # CASE A: Position 0 (Center)
    # Neighbors: -1 (Val=1), 1 (Val=1). Own Val=0.
    # Edge (-1 -> 0): a=1, b=0 -> OneNull -> a+2 -> 3
    # Edge ( 1 -> 0): a=1, b=0 -> OneNull -> a+2 -> 3
    # Step Sum: 3 + 3 = 6
    val_0 = get_res(0)
    print(f"Pos 0 (Expected 6.0): {val_0}")
    assert val_0 == 6.0

    # CASE B: Position -1 (Has Value)
    # Neighbors: -2 (Val=0), 0 (Val=0). Own Val=1.
    # Edge (-2 -> -1): a=0, b=1 -> OneNull -> b+2 -> 3
    # Edge ( 0 -> -1): a=0, b=1 -> OneNull -> b+2 -> 3
    # Step Sum: 3 + 3 = 6
    val_neg1 = get_res(-1)
    print(f"Pos -1 (Expected 6.0): {val_neg1}")
    assert val_neg1 == 6.0

    # CASE C: Position -3 (Boundary/Empty)
    # Neighbors: -4 (Invalid/None), -2 (Val=0). Own Val=0.
    # Note: Topology usually filters invalid neighbors. So only -2 exists.
    # Edge (-2 -> -3): a=0, b=0 -> BothNull -> 3
    # Step Sum: 3
    val_neg3 = get_res(-3)
    print(f"Pos -3 (Expected 3.0): {val_neg3}")
    assert val_neg3 == 3.0

    print("✅ All Logic Rules Verified!")


if __name__ == "__main__":
    test_bizarre_composition_manual()