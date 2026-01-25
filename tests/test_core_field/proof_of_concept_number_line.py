import jax.numpy as jnp
import numpy as np
import pytest
from typing import Any, Sequence

from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.core.state import AbstractState, AbstractDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
# NEW IMPORT
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposer


# =========================================================
# 1. SETUP: ALGEBRA & TOPOLOGY
# =========================================================

class RealFieldAlgebra(IFieldAlgebra):
    """Simple Scalar Algebra (Standard Math)"""
    dim = 1
    dtype = jnp.float32

    def add(self, a, b): return a + b

    def mul(self, a, b): return a * b

    def get_unity(self, shape): return jnp.ones(shape, dtype=self.dtype)

    def get_zero(self, shape): return jnp.zeros(shape, dtype=self.dtype)


class WalkerTopology(DiscreteTopology):
    """
    1D Walker: Moves Left (-1), Stays (0), Moves Right (+1)
    """

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        # Input: Tuple (x,) -> Output: [(x-1,), (x,), (x+1,)]
        x = state_val[0]
        return [(x - 1,), (x,), (x + 1,)]


# =========================================================
# 2. THE TEST
# =========================================================

def test_single_step_relocation_with_composition():
    print("\n=========================================================")
    print("🧪 TEST: Single Step Relocation (via Composition)")
    print("=========================================================")

    # --- A. INITIALIZATION ---
    space = AbstractDiscreteStateSpace([AbstractState("dummy", {})])
    topology = WalkerTopology(space)
    algebra = RealFieldAlgebra()
    mapper = DiscreteFieldMapper(space, algebra)

    # --- B. SCOUT PHASE ---
    initial_states = [(0,), (1,)]
    print(f"-> Scouting neighbors for: {initial_states}")

    topology.get_raw_successor(initial_states)
    topology._raw_sync_to_device()
    matrix = topology._raw_jax_matrix

    print(f"-> Matrix Built. Shape: {matrix.shape}")

    # --- C. SYNC MAPPER ---
    num_states = matrix.shape[0]
    mapper.sync_size(num_states)

    # --- D. COMPOSE FIELDS (The New Part) ---
    # Goal: Combine Field A (Impulse at 0) and Field B (Impulse at 1)

    # 1. Get IDs
    id_0 = topology._raw_to_id[(0,)]
    id_1 = topology._raw_to_id[(1,)]
    print(f"-> ID Mapping: (0,)={id_0}, (1,)={id_1}")

    # 2. Define Sparse Inputs (Mocking two independent sources)
    # Source A: value 1.0 at index id_0
    ids_a = jnp.array([[id_0]])
    vals_a = jnp.array([[1.0]])

    # Source B: value 1.0 at index id_1
    ids_b = jnp.array([[id_1]])
    vals_b = jnp.array([[1.0]])

    class AdditionCompositions(AdditionComposition):
        def compose(self, a, b): return a + b

        def get_identity(self, shape, dtype): return jnp.zeros(shape, dtype=dtype)

    print("-> Composing Sparse Fields via FieldSpaceComposer...")

    # 3. Call The Raw Composer
    # This merges the two sparse lists into one unique list
    comp_ids, comp_vals = FieldSpaceComposer.compose_raw(
        ids_a, vals_a,
        ids_b, vals_b,
         AdditionCompositions()
    )

    print(f"   Composed IDs: {comp_ids.flatten()}")
    print(f"   Composed Vals: {comp_vals.flatten()}")

    # 4. Scatter into Dense Vector (Physics Prep)
    # We start with empty universe and fill in the composed active sites
    current_field = jnp.zeros((num_states, 1), dtype=jnp.float32)

    # JAX syntax: at[indices].set(values)
    # Note: comp_ids might need flattening depending on shape, usually (N, 1) or (N,)
    # We ensure indices are integers for indexing
    current_field = current_field.at[comp_ids.flatten().astype(int)].set(comp_vals)

    mapper.apply_vector(current_field)
    print(f"-> Initial Dense Field:\n{current_field.flatten()}")

    # --- E. SOLVE (Physics Step) ---
    print("-> Running Physics Step...")
    next_field = matrix.T @ current_field
    mapper.apply_vector(next_field)

    # --- F. VERIFICATION ---
    id_neg1 = topology._raw_to_id[(-1,)]
    id_2 = topology._raw_to_id[(2,)]

    val_neg1 = float(next_field[id_neg1][0])
    val_0 = float(next_field[id_0][0])
    val_1 = float(next_field[id_1][0])
    val_2 = float(next_field[id_2][0])

    print("\n--- Results ---")
    print(f"  State (-1): {val_neg1} (Expected 1.0)")
    print(f"  State ( 0): {val_0}    (Expected 2.0)")
    print(f"  State ( 1): {val_1}    (Expected 2.0)")
    print(f"  State ( 2): {val_2}    (Expected 1.0)")

    assert val_neg1 == 1.0
    assert val_0 == 2.0
    assert val_1 == 2.0
    assert val_2 == 1.0

    print("✅ SUCCESS: Composition + Diffusion logic is correct.")


if __name__ == "__main__":
    test_single_step_relocation_with_composition()