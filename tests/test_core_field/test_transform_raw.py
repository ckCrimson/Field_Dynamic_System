import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.transform import LinearTransform, NonLinearTransform, NormTransform
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransformer


def test_discrete_raw_kernels():
    print("\n=== TEST: Raw Field Transform Kernels ===")

    # 1. Setup Raw Data (Batch of 3 Vectors, Dim=2)
    # V1=[1,0], V2=[0,2], V3=[3,4]
    raw_buffer = jnp.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [3.0, 4.0]
    ])

    print(f"  Input Shape: {raw_buffer.shape}")

    # ---------------------------------------------------------
    # Case A: Linear Transform (Matrix Multiplication)
    # Scale by 10. Matrix = [[10, 0], [0, 10]]
    # ---------------------------------------------------------
    M = jnp.eye(2) * 10.0
    op_lin = LinearTransform(M)

    res_lin = FieldSpaceTransformer.apply_raw(raw_buffer, op_lin)

    expected_lin = jnp.array([
        [10.0, 0.0],
        [0.0, 20.0],
        [30.0, 40.0]
    ])

    assert jnp.allclose(res_lin, expected_lin)
    print("  ✅ Linear Transform (Raw): Passed")

    # ---------------------------------------------------------
    # Case B: Norm Transform (Vector -> Scalar)
    # Should reduce (N, 2) -> (N, 1)
    # Norms: 1.0, 2.0, 5.0
    # ---------------------------------------------------------
    op_norm = NormTransform()

    res_norm = FieldSpaceTransformer.apply_raw(raw_buffer, op_norm)

    expected_norm = jnp.array([
        [1.0],
        [2.0],
        [5.0]
    ])

    assert jnp.allclose(res_norm, expected_norm)
    assert res_norm.shape == (3, 1)
    print("  ✅ Norm Transform (Raw): Passed")

    # ---------------------------------------------------------
    # Case C: Non-Linear Transform (Element-wise Square)
    # ---------------------------------------------------------
    op_sq = NonLinearTransform(lambda x: x ** 2)

    res_sq = FieldSpaceTransformer.apply_raw(raw_buffer, op_sq)

    expected_sq = jnp.array([
        [1.0, 0.0],
        [0.0, 4.0],
        [9.0, 16.0]
    ])

    assert jnp.allclose(res_sq, expected_sq)
    print("  ✅ Non-Linear Transform (Raw): Passed")


if __name__ == "__main__":
    test_discrete_raw_kernels()