import jnp
import pytest
import numpy as np
from abc import ABC, abstractmethod

from src.field_dynamic_system.core.field.compositions import AdditionComposition, MultiplicationComposition
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteStateSpace
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import ElementwiseKernel
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from tests.test_core_field.proof_of_concept_number_line import RealFieldAlgebra
from tests.test_generator.test_manual_verification import L1Normalizer


# --- DEFINE COMPOSERS WITH IDENTITY ---

# --- SETUP ---
raw_states = list(range(-5, 6))
space = DiscreteStateSpace(raw_states)
algebra = RealFieldAlgebra()
field_mapper = DiscreteFieldMapper.impulse(space, algebra, 0)


class OneDimWalkerTopology(DiscreteTopology):
    def compute_neighbors(self, state_0):
        return [state_0 - 1, state_0 + 1]


class OneDimWalkerKernel(ElementwiseKernel):
    def compute_transition_value(self, state_in, state_out):
        return 1.0


def test_walker_survives_multiplication():
    print("\n==================================================")
    print("🚶 WALKER TEST: Identity Logic Check")
    print("==================================================")

    topology = OneDimWalkerTopology(space)

    # We use MULTIPLICATION for Chain.
    # Old Logic: Incoming(0.5) * Target(0.0) = 0.0 (Death)
    # New Logic: Incoming(0.5) * Identity(1.0) = 0.5 (Life)
    generator = GenericMarkovianDiscreteFieldGenerator(
        topology=topology,
        kernel=OneDimWalkerKernel(),
        step_composer=AdditionComposition(),
        chain_composer=MultiplicationComposition(),  # <--- TESTING THIS
        global_composer=None,
        intrinsic_transform=L1Normalizer(),
        extrinsic_transform=L1Normalizer()
    )

    result = generator.generate_multi_step(field_mapper, steps=1)

    # Check Neighbors
    def get_val(v):
        idx = space.get_index_of(v)
        return float(result.raw_buffer[idx][0])

    val_neg = get_val(-1)
    val_pos = get_val(1)

    print(f"-> Neighbor -1: {val_neg}")
    print(f"-> Neighbor +1: {val_pos}")

    assert val_neg > 0.0, "Walker died! Identity substitution failed."
    assert val_pos > 0.0, "Walker died! Identity substitution failed."

    print("✅ SUCCESS: MultiplicationComposition correctly used Identity(1.0) for empty space.")


if __name__ == "__main__":
    test_walker_survives_multiplication()