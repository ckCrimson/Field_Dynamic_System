import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra, VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.core.field.transform import LinearTransform, NonLinearTransform, NormTransform
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransformer


# =========================================================================
# TEST 1: Single Field Value Transform (Real)
# =========================================================================
def test_1_single_value_real():
    print("\n=== TEST 1: Single Real Value Transforms ===")

    # Setup
    val = jnp.array([3.0])  # Scalar 3.0

    # A. Non-Linear: x^2 -> 9.0
    op_sq = NonLinearTransform(lambda x: x ** 2, RealFieldAlgebra)
    res_sq = op_sq(val)
    assert jnp.allclose(res_sq, 9.0)
    print("  ✅ Non-Linear (x^2): Passed")

    # B. Affine (User called it Linear): x + 2 -> 5.0
    op_add = NonLinearTransform(lambda x: x + 2.0, RealFieldAlgebra)
    res_add = op_add(val)
    assert jnp.allclose(res_add, 5.0)
    print("  ✅ Affine (x+2): Passed")

    # C. Norm: |x| -> 3.0
    op_norm = NormTransform()
    res_norm = op_norm(val)
    assert jnp.allclose(res_norm, 3.0)

    # Check negative norm
    res_neg_norm = op_norm(jnp.array([-3.0]))
    assert jnp.allclose(res_neg_norm, 3.0)
    print("  ✅ Norm (|x|): Passed")


# =========================================================================
# TEST 2: Single Field Value Transform (Vector)
# =========================================================================
def test_2_single_value_vector():
    print("\n=== TEST 2: Single Vector Value Transforms ===")

    # Setup: Vector v = [1, 2, 3]
    v_raw = jnp.array([1.0, 2.0, 3.0])

    # A. Linear (Matrix Scaling): 2*Identity * v -> [2, 4, 6]
    # Matrix: [[2,0,0], [0,2,0], [0,0,2]]
    M = jnp.eye(3) * 2.0
    op_scale = LinearTransform(M, VectorFieldAlgebra)
    res_scale = op_scale(v_raw)

    assert jnp.allclose(res_scale, jnp.array([2.0, 4.0, 6.0]))
    print("  ✅ Linear (2*I): Passed")

    # B. Vector Norm: ||v|| -> sqrt(1+4+9) = sqrt(14) ~= 3.74
    op_norm = NormTransform()
    res_norm = op_norm(v_raw)

    expected = jnp.sqrt(14.0)
    assert jnp.allclose(res_norm, expected)
    # Check shape: Should be (1,) scalar
    assert res_norm.shape == (1,)
    print("  ✅ Vector Norm: Passed")


# =========================================================================
# TEST 3: Discrete Field Space Transform
# =========================================================================
def test_3_discrete_space_transform():
    print("\n=== TEST 3: Discrete Space Transforms ===")

    # Setup: 2 States
    s1 = VectorState((0, 0, 0))
    s2 = VectorState((1, 0, 0))
    space = VectorStateSpace([s1, s2], dim=3)

    # Field: s1=[1,1,1], s2=[2,0,0]
    alg_vec = VectorFieldAlgebra(dim=3)
    mapper = DiscreteFieldMapper(space, alg_vec)
    mapper.set_value_at(s1, [1., 1., 1.])
    mapper.set_value_at(s2, [2., 0., 0.])

    # --- Case A: Linear Transform (Scaling by 2) ---
    M = jnp.eye(3) * 2.0
    op_lin = LinearTransform(M, VectorFieldAlgebra)

    res_mapper_lin = FieldSpaceTransformer.apply(mapper, op_lin)

    # Verify s1 -> [2,2,2]
    val_s1 = res_mapper_lin.get_fields_at(s1)[0].value
    assert jnp.allclose(val_s1, jnp.array([2., 2., 2.]))
    print("  ✅ Linear Transform (Space): Passed")

    # --- Case B: Norm Transform (Vector -> Scalar) ---
    op_norm = NormTransform()
    res_mapper_norm = FieldSpaceTransformer.apply(mapper, op_norm)

    # Verify s1 -> sqrt(3), s2 -> 2.0
    val_s1_norm = res_mapper_norm.get_fields_at(s1)[0].value
    val_s2_norm = res_mapper_norm.get_fields_at(s2)[0].value

    assert jnp.allclose(val_s1_norm, jnp.sqrt(3.0))
    assert jnp.allclose(val_s2_norm, 2.0)

    # Verify Algebra Change: Output should be RealFieldAlgebra (Dim 1)
    assert isinstance(res_mapper_norm.algebra, RealFieldAlgebra)
    print("  ✅ Norm Transform (Space & Algebra Switch): Passed")


# =========================================================================
# TEST 4: Continuous Field Space Transform
# =========================================================================
def test_4_continuous_space_transform():
    print("\n=== TEST 4: Continuous Space Transforms ===")

    # Setup: 1D Line, Background f(x) = x
    # We use Abstract/Hypercube mock for simplicity
    class LineSpace(AbstractDiscreteStateSpace):
        # Minimal Mock for continuous test
        pass

    space = LineSpace([])
    alg_real = RealFieldAlgebra()

    # Background: f(x) = x
    def bg_identity(x): return x

    mapper = ContinuousFieldMapper(space, alg_real, bg_func=bg_identity)

    # --- Case A: Non-Linear Transform (Square) ---
    # New Field should be f(x) = x^2
    op_sq = NonLinearTransform(lambda x: x ** 2, RealFieldAlgebra)

    res_mapper_sq = FieldSpaceTransformer.apply(mapper, op_sq)

    # Verify at arbitrary point x=3.0 -> 9.0
    # Note: Continuous mapper expects state objects usually,
    # but here we pass raw value to test the internal bg_func if needed,
    # OR we pass a dummy state if the BG func handles it.
    # Let's assume input state values are passed through.

    query_val = jnp.array([[3.0]])  # Shape (1, 1)
    # Direct check of the composed function to avoid State wrapping issues in test
    res_val = res_mapper_sq.background_func(query_val)

    assert jnp.allclose(res_val, 9.0)
    print("  ✅ Continuous Lazy Composition (x^2): Passed")


if __name__ == "__main__":
    test_1_single_value_real()
    test_2_single_value_vector()
    test_3_discrete_space_transform()
    test_4_continuous_space_transform()