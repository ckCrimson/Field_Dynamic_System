import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.compositions import AdditionComposition, MultiplicationComposition
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposer


# =========================================================================
# TEST 1: Aligned Composition (Fast Path)
# Use Case: Simple Vector Addition A + B
# =========================================================================
def test_raw_aligned_composition():
    print("\n=== TEST 1: Raw Aligned Composition (A + B) ===")

    # Setup: 3 states, 2D vectors
    buf_a = jnp.array([[1., 1.], [2., 2.], [3., 3.]])
    buf_b = jnp.array([[10., 0.], [0., 10.], [5., 5.]])

    op = AdditionComposition()

    # Execute Raw Kernel
    res = FieldSpaceComposer.compose_raw(buf_a, buf_b, op)

    expected = jnp.array([[11., 1.], [2., 12.], [8., 8.]])

    assert jnp.allclose(res, expected)
    print("  ✅ Aligned Addition Passed")


# =========================================================================
# TEST 2: Unaligned Composition (Scatter-Gather)
# Use Case: Merging two partial fields into a Global Space
# =========================================================================
def test_raw_unaligned_composition():
    print("\n=== TEST 2: Raw Unaligned Composition (Intersection & Union) ===")

    # Scenario:
    # Global Space has 5 slots: [0, 1, 2, 3, 4]
    TOTAL_SIZE = 5

    # Field A exists at indices [0, 2, 4]
    # Values: [1,1], [2,2], [4,4]
    indices_a = jnp.array([0, 2, 4])
    buf_a = jnp.array([[1., 1.], [2., 2.], [4., 4.]])

    # Field B exists at indices [2, 3] (Overlap at 2)
    # Values: [10,10], [30,30]
    indices_b = jnp.array([2, 3])
    buf_b = jnp.array([[10., 10.], [30., 30.]])

    op = AdditionComposition()

    # Execute Raw Unaligned Kernel
    # Logic:
    # Idx 0: A(1,1) + 0      = [1, 1]
    # Idx 1: 0 + 0           = [0, 0]  (Empty)
    # Idx 2: A(2,2) + B(10,10) = [12, 12] (Intersection)
    # Idx 3: 0 + B(30,30)    = [30, 30]
    # Idx 4: A(4,4) + 0      = [4, 4]

    res = FieldSpaceComposer.compose_unaligned_raw(
        buf_a, indices_a,
        buf_b, indices_b,
        TOTAL_SIZE,
        op
    )

    # Verification
    print(f"  Result Buffer:\n{res}")

    expected = jnp.array([
        [1., 1.],
        [0., 0.],
        [12., 12.],
        [30., 30.],
        [4., 4.]
    ])

    assert jnp.allclose(res, expected)
    print("  ✅ Unaligned Scatter-Add Passed")


# =========================================================================
# TEST 3: Unaligned Generic Operation (Multiplication)
# Use Case: Checking the 'else' block for Read-Modify-Write
# =========================================================================
def test_raw_unaligned_multiplication():
    print("\n=== TEST 3: Raw Unaligned Multiplication ===")

    # Scenario: Overlap multiplication
    # Global Size = 3
    TOTAL_SIZE = 3

    # Field A at [0, 1]: Values [2], [3]
    indices_a = jnp.array([0, 1])
    buf_a = jnp.array([[2.], [3.]])

    # Field B at [1, 2]: Values [4], [5]
    indices_b = jnp.array([1, 2])
    buf_b = jnp.array([[4.], [5.]])

    op = MultiplicationComposition()

    # NOTE: The current `compose_unaligned_raw` initializes the global buffer with ZEROS.
    # For Multiplication, this is tricky.
    # - Idx 0: A(2) * Init(0) -> 0? Or just A(2)?
    # - Idx 1: A(3) * B(4) -> 12.
    # - Idx 2: Init(0) * B(5) -> 0?

    # The current implementation executes:
    # 1. Init Global = [0, 0, 0]
    # 2. Set A: Global = [2, 3, 0]
    # 3. Modify B: Global[1] = Global[1] * B[0] = 3 * 4 = 12
    #              Global[2] = Global[2] * B[1] = 0 * 5 = 0

    # This behavior is correct for "Scatter-Update" logic on a zero-initialized canvas.
    # If we wanted Identity initialization (1s), the kernel would need to know the identity element.
    # For now, we test the CURRENT behavior (Zero Init).

    res = FieldSpaceComposer.compose_unaligned_raw(
        buf_a, indices_a,
        buf_b, indices_b,
        TOTAL_SIZE,
        op
    )

    print(f"  Result (Mul): \n{res}")

    expected = jnp.array([
        [2.],  # Set from A
        [12.],  # 3 * 4
        [0.]  # 0 * 5 (Since canvas was 0)
    ])

    assert jnp.allclose(res, expected)
    print("  ✅ Generic Read-Modify-Write Passed")


if __name__ == "__main__":
    test_raw_aligned_composition()
    test_raw_unaligned_composition()
    test_raw_unaligned_multiplication()