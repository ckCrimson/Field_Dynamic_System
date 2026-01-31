import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import ops

from src.field_dynamic_system.core.field.transform import FieldTransform
# --- IMPORTS FROM YOUR REAL SYSTEM ---
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.neighbor.discrete import GraphTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import UnbiasedKernel
from src.field_dynamic_system.core.field.compositions import FieldComposition, AdditionComposition


# --- CUSTOM LOGIC DEFINITIONS (Because they are specific to this test scenario) ---

class MultiplicationComposition(FieldComposition):
    def compose(self, a, b):
        return a * b


class DegreeNormalizer(FieldTransform):
    """Intrinsic: Divides by 2 (Specific to this test's physics)"""

    def __call__(self, val):
        return val / 2.0


class L1Normalizer(FieldTransform):
    """Extrinsic: Normalizes sum to 1.0"""

    def __call__(self, field):
        total = jnp.sum(field)
        return field / jnp.where(total == 0, 1.0, total)


# --- THE REAL TEST ---

def test_real_physics_integration():
    print("\n==================================================")
    print("🔥 REAL INTEGRATION TEST: 1D Multiplicative Walker")
    print("==================================================")

    # 1. REAL STATE SPACE
    # 5 discrete integer states: 0, 1, 2, 3, 4
    states = [0, 1, 2, 3, 4]
    space = DiscreteStateSpace(states)

    # 2. REAL TOPOLOGY
    # Explicit edges for a line graph: 0-1-2-3-4
    edges = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
        (3, 4), (4, 3)
    ]
    # We use your actual GraphTopology implementation
    topology = GraphTopology(space, edges=edges)

    # 3. REAL FIELD MAPPERS
    # Algebra: 1D Real Numbers
    algebra = RealFieldAlgebra()

    # A. Initial State Mapper
    # [0.1, 0.2, 0.4, 0.2, 0.1]
    start_mapper = DiscreteFieldMapper(space, algebra)
    start_mapper.set_value_at(0, 0.1)
    start_mapper.set_value_at(1, 0.2)
    start_mapper.set_value_at(2, 0.4)
    start_mapper.set_value_at(3, 0.2)
    start_mapper.set_value_at(4, 0.1)

    # B. Global Field Mapper
    # [2.0, 2.0, 2.0, 2.0, 2.0]
    global_mapper = DiscreteFieldMapper(space, algebra)
    for s in states:
        global_mapper.set_value_at(s, 2.0)

    # 4. REAL GENERATOR CONFIGURATION
    # Using the exact class we just wrote, passing real components.
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=topology,
        kernel=UnbiasedKernel(prob=1.0),  # Real Kernel (returns 1.0 weights)
        step_composer=AdditionComposition(),  # Sum neighbors
        chain_composer=MultiplicationComposition(),  # Incoming * Old_Target
        global_composer=MultiplicationComposition(),  # Incoming * Global
        intrinsic_transform=DegreeNormalizer(),  # Divide by 2
        extrinsic_transform=L1Normalizer()  # Normalize Sum
    )

    # 5. EXECUTION (JAX)
    print("-> Running Generator (Steps=1)...")
    result_mapper = gen.generate_multi_step(start_mapper, steps=1, global_mapper=global_mapper)

    # Get Data
    jax_result = np.array(result_mapper.raw_buffer.flatten())
    print(f"-> JAX Result: {jax_result}")

    # 6. MANUAL VERIFICATION (The Truth)
    # We calculate expected values by hand based on the graph structure
    # State: [0.1, 0.2, 0.4, 0.2, 0.1]

    # Node 0 (Target): Receives from 1
    #   Src(1)=0.2 -> Global(*2)=0.4 -> Intrinsic(/2)=0.2 -> Weight(*1)=0.2 -> Chain(*Old_Tgt_0)=0.2*0.1 = 0.02
    #   Step Sum = 0.02

    # Node 1 (Target): Receives from 0 and 2
    #   From 0: Src(0)=0.1 -> *2 -> /2 -> *1 -> Chain(*Old_Tgt_1)=0.1*0.2 = 0.02
    #   From 2: Src(2)=0.4 -> *2 -> /2 -> *1 -> Chain(*Old_Tgt_1)=0.4*0.2 = 0.08
    #   Step Sum = 0.10

    # Node 2 (Target): Receives from 1 and 3
    #   From 1: Src(1)=0.2 -> ... -> Chain(*Old_Tgt_2)=0.2*0.4 = 0.08
    #   From 3: Src(3)=0.2 -> ... -> Chain(*Old_Tgt_2)=0.2*0.4 = 0.08
    #   Step Sum = 0.16

    # Node 3 (Target): Receives from 2 and 4
    #   From 2: Src(2)=0.4 -> ... -> Chain(*Old_Tgt_3)=0.4*0.2 = 0.08
    #   From 4: Src(4)=0.1 -> ... -> Chain(*Old_Tgt_3)=0.1*0.2 = 0.02
    #   Step Sum = 0.10

    # Node 4 (Target): Receives from 3
    #   From 3: Src(3)=0.2 -> ... -> Chain(*Old_Tgt_4)=0.2*0.1 = 0.02
    #   Step Sum = 0.02

    # Raw Vector: [0.02, 0.10, 0.16, 0.10, 0.02]
    # Sum = 0.40

    # Extrinsic Normalization:
    # [0.02/0.4, 0.10/0.4, 0.16/0.4, 0.10/0.4, 0.02/0.4]
    # [0.05, 0.25, 0.40, 0.25, 0.05]

    manual_result = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
    print(f"-> Manual Exp: {manual_result}")

    # 7. ASSERTION
    diff = np.abs(jax_result - manual_result)
    print(f"-> Max Diff:   {np.max(diff):.8f}")

    assert np.allclose(jax_result, manual_result, atol=1e-6)
    print("✅ SUCCESS: Real implementation matches Physics perfectly.")


if __name__ == "__main__":
    test_real_physics_integration()