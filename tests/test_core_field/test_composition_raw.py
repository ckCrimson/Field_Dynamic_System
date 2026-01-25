import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposer
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace


# =========================================================
# 1. TOPOLOGY (King's Grid)
# =========================================================
class KingsGraphTopology:
    def __init__(self):
        self.offsets = jnp.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ])

    def get_raw_successor(self, current_coords: jnp.ndarray) -> jnp.ndarray:
        neighbors = current_coords[:, None, :] + self.offsets[None, :, :]
        return neighbors.reshape(-1, 2)


# =========================================================
# 2. ROBUST INTEGRATION TEST
# =========================================================
def test_full_round_trip_robust():
    print("\n=== TEST: Full Round Trip (Robust String States) ===")

    # A. INITIALIZATION
    topology = KingsGraphTopology()
    algebra = RealFieldAlgebra()

    state_a = jnp.array([[0, 0]])
    state_b = jnp.array([[0, 1]])

    # B. RAW SIMULATION
    next_coords_a = topology.get_raw_successor(state_a)
    next_coords_b = topology.get_raw_successor(state_b)

    vals_a = jnp.ones((len(next_coords_a), 1))
    vals_b = jnp.ones((len(next_coords_b), 1))

    # C. RAW COMPOSITION
    print("-> Running Raw Composition Kernel...")
    op = AdditionComposition()
    unique_coords, merged_vals = FieldSpaceComposer.compose_raw(
        next_coords_a, vals_a,
        next_coords_b, vals_b,
        op
    )
    unique_coords.block_until_ready()

    # D. RECONSTRUCTION (With String Conversion)
    print("-> Reconstructing FieldMapper (Coords -> Strings)...")

    # 1. Robust Conversion: Coordinate -> String Key "-1,0"
    # This eliminates all ambiguity about Int32 vs Int64 vs Python Int.
    def to_key(row):
        return f"{int(row[0])},{int(row[1])}"

    raw_state_keys = [to_key(r) for r in np.array(unique_coords)]
    raw_values = np.array(merged_vals)

    # 2. Create Space
    final_space = AbstractDiscreteStateSpace(raw_state_keys)

    # 3. Alignment (Scatter)
    N = len(raw_state_keys)
    dim = merged_vals.shape[1]

    aligned_buffer = np.zeros((N, dim), dtype=float)
    aligned_mask = np.zeros(N, dtype=bool)

    # Force build a fresh lookup map
    state_to_idx = {s: i for i, s in enumerate(final_space.states)}

    print("-> Debug: Alignment Dump")
    for i, state_key in enumerate(raw_state_keys):
        target_idx = state_to_idx[state_key]
        val = raw_values[i]

        # Write to buffer
        aligned_buffer[target_idx] = val
        aligned_mask[target_idx] = True

        # DEBUG: Print critical states
        if state_key == "-1,0":
            print(f"   Writing '-1,0': Value={val} -> Index={target_idx}")

    # 4. Create Mapper
    final_mapper = DiscreteFieldMapper(
        final_space,
        algebra,
        explicit_buffer=jnp.array(aligned_buffer),
        mask_buffer=jnp.array(aligned_mask)
    )

    # E. VERIFICATION
    print("-> Verifying Logic...")

    # Logic:
    # A(0,0) reaches (-1,0). B(0,1) reaches (-1,0). Sum = 2.0.

    # Targets (Strings)
    target_overlap = "-1,0"

    # 1. Verify Space Indexing
    idx = final_mapper.state_space.get_index_of(target_overlap)
    print(f"   Lookup '{target_overlap}' -> Index {idx}")
    assert idx is not None, f"Space failed to index {target_overlap}"

    # 2. Verify Buffer Content Directly
    buffer_val = final_mapper.explicit_buffer[idx]
    print(f"   Buffer[{idx}] = {buffer_val}")

    # 3. Verify Object API
    field_objs = final_mapper.get_fields_at(target_overlap)
    val = float(field_objs[0].value.item())
    print(f"   API Value: {val}")

    assert val == 2.0, f"Expected 2.0 for overlap state {target_overlap}, got {val}"

    # Verify a unique state (e.g. A's bottom-left corner)
    target_unique = "-1,-1"
    val_unique = float(final_mapper.get_fields_at(target_unique)[0].value.item())
    assert val_unique == 1.0, f"Expected 1.0 for unique state {target_unique}, got {val_unique}"

    print(f"✅ PASSED: Full round trip successful with string keys.")


if __name__ == "__main__":
    test_full_round_trip_robust()