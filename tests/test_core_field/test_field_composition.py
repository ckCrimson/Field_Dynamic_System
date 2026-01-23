import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, VectorStateSpace
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra, VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.core.field.compositions import AdditionComposition, InnerProductComposition
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceComposer


# =========================================================================
# TEST 1: Unaligned - Overlapping Spaces (Symbolic Join)
# Case: Space A=[A, B], Space B=[B, C]
# =========================================================================
def test_discrete_unaligned_overlap():
    print("\n=== TEST 1: Discrete Unaligned (Overlap A,B vs B,C) ===")

    # 1. Setup DISTINCT Space Objects
    space_a = AbstractDiscreteStateSpace(["StateA", "StateB"])
    space_b = AbstractDiscreteStateSpace(["StateB", "StateC"])  # Overlap on "StateB"

    alg = VectorFieldAlgebra(dim=2)  # 2D Vectors

    # Field 1: A=[1, 1], B=[2, 2]
    f1 = DiscreteFieldMapper(space_a, alg)
    f1.set_value_at("StateA", [1., 1.])
    f1.set_value_at("StateB", [2., 2.])

    # Field 2: B=[10, 10], C=[20, 20]
    f2 = DiscreteFieldMapper(space_b, alg)
    f2.set_value_at("StateB", [10., 10.])
    f2.set_value_at("StateC", [20., 20.])

    # Compose: F3 = F1 + F2
    # Logic:
    # StateA: Only in F1 -> [1, 1] + 0 = [1, 1]
    # StateB: Intersection -> [2, 2] + [10, 10] = [12, 12]
    # StateC: Only in F2 -> 0 + [20, 20] = [20, 20]

    f3 = FieldSpaceComposer.compose(
        f1, f2,
        composition_op=AdditionComposition(),
        output_algebra=alg
    )

    # Verify StateA
    res_a = f3.get_fields_at("StateA")[0].value
    assert jnp.allclose(res_a, jnp.array([1., 1.]))

    # Verify StateB (The Intersection)
    res_b = f3.get_fields_at("StateB")[0].value
    assert jnp.allclose(res_b, jnp.array([12., 12.]))

    # Verify StateC
    res_c = f3.get_fields_at("StateC")[0].value
    assert jnp.allclose(res_c, jnp.array([20., 20.]))

    # Verify Space Size: Should be 3 (Union of 2+2 with 1 overlap)
    assert f3.state_space.num_states == 3
    print("  ✅ Overlapping Symbolic Join: Passed")


# =========================================================================
# TEST 2: Unaligned - Reordered Indices
# Case: Space A=[X, Y], Space B=[Y, X]
# This breaks naive index addition.
# =========================================================================
def test_discrete_unaligned_reorder():
    print("\n=== TEST 2: Discrete Unaligned (Reordered X,Y vs Y,X) ===")

    space_a = AbstractDiscreteStateSpace(["X", "Y"])
    space_b = AbstractDiscreteStateSpace(["Y", "X"])  # Same states, swapped index

    alg = RealFieldAlgebra()

    # Field 1 (Ordered X, Y): X=10, Y=20
    f1 = DiscreteFieldMapper(space_a, alg)
    f1.set_value_at("X", 10.0)
    f1.set_value_at("Y", 20.0)

    # Field 2 (Ordered Y, X): Y=5, X=5
    f2 = DiscreteFieldMapper(space_b, alg)
    f2.set_value_at("Y", 5.0)
    f2.set_value_at("X", 5.0)

    # Compose: F3 = F1 + F2
    f3 = FieldSpaceComposer.compose(f1, f2, AdditionComposition(), alg)

    # Verify X: 10 + 5 = 15
    res_x = f3.get_fields_at("X")[0].value
    assert jnp.allclose(res_x, 15.0)

    # Verify Y: 20 + 5 = 25
    res_y = f3.get_fields_at("Y")[0].value
    assert jnp.allclose(res_y, 25.0)

    print("  ✅ Reordered Index Matching: Passed")


# =========================================================================
# TEST 3: Unaligned - Subset + Background
# Case: Space A=[1, 2, 3] (Explicit), Space B=[2] (Explicit + Background)
# =========================================================================
def test_discrete_subset_with_bg():
    print("\n=== TEST 3: Subset with Background Integration ===")

    # Space A has 3 states
    space_a = AbstractDiscreteStateSpace(["S1", "S2", "S3"])

    # Space B has only "S2", but has a Background Function of 100.0
    space_b = AbstractDiscreteStateSpace(["S2"])

    alg = RealFieldAlgebra()

    # Field A: S1=1, S2=2, S3=3
    f1 = DiscreteFieldMapper(space_a, alg)
    f1.set_value_at("S1", 1.0)
    f1.set_value_at("S2", 2.0)
    f1.set_value_at("S3", 3.0)

    # Field B: S2=50 (Explicit). Background = 100.0
    # Note: Explicit overrides Background usually, but let's check composition logic.
    # Our logic uses Explicit if available, else Background.
    # So for B: S2 is 50. S1, S3 are 100 (via BG).

    bg_func_b = lambda s: jnp.full((len(s), 1), 100.0)
    f2 = DiscreteFieldMapper(space_b, alg, bg_func=bg_func_b)
    f2.set_value_at("S2", 50.0)

    # Compose: F3 = F1 + F2
    f3 = FieldSpaceComposer.compose(f1, f2, AdditionComposition(), alg)

    # Verify S1 (Only in A explicitly)
    # Logic: A(Explicit=1) + B(Implicit via BG=100) = 101
    res_s1 = f3.get_fields_at("S1")[0].value
    assert jnp.allclose(res_s1, 101.0)

    # Verify S2 (Intersection)
    # Logic: A(Explicit=2) + B(Explicit=50) = 52
    res_s2 = f3.get_fields_at("S2")[0].value
    assert jnp.allclose(res_s2, 52.0)

    # Verify S3 (Only in A explicitly)
    # Logic: A(Explicit=3) + B(Implicit via BG=100) = 103
    res_s3 = f3.get_fields_at("S3")[0].value
    assert jnp.allclose(res_s3, 103.0)

    print("  ✅ Implicit Value Fetching (Background): Passed")


# =========================================================================
# TEST 4: Continuous + Continuous (Unchanged Logic)
# =========================================================================
def test_continuous_continuous():
    print("\n=== TEST 4: Continuous + Continuous ===")

    class MockSpace(AbstractDiscreteStateSpace): pass

    space = MockSpace([])
    alg = RealFieldAlgebra()

    # F1: x, F2: 2x
    f1 = ContinuousFieldMapper(space, alg, bg_func=lambda x: x)
    f2 = ContinuousFieldMapper(space, alg, bg_func=lambda x: 2 * x)

    f3 = FieldSpaceComposer.compose(f1, f2, AdditionComposition(), alg)

    val = jnp.array([[10.0]])
    res = f3.background_func(val)
    assert jnp.allclose(res, 30.0)
    print("  ✅ Continuous Composition: Passed")


# =========================================================================
# TEST 5: Hybrid (Discrete + Continuous)
# =========================================================================
def test_hybrid_composition():
    print("\n=== TEST 5: Hybrid (Discrete + Continuous) ===")
    space = AbstractDiscreteStateSpace(["D"])
    alg = RealFieldAlgebra()

    f_disc = DiscreteFieldMapper(space, alg)
    f_disc.set_value_at("D", 10.0)

    f_cont = ContinuousFieldMapper(space, alg, bg_func=lambda x: jnp.full((len(x), 1), 1.0))

    f_hybrid = FieldSpaceComposer.compose(f_disc, f_cont, AdditionComposition(), alg)

    # Query via object wrapper (cheating slightly by accessing first element)
    res = f_hybrid.get_fields_at("D")[0].value
    assert jnp.allclose(res, 11.0)
    print("  ✅ Hybrid Composition: Passed")


if __name__ == "__main__":
    test_discrete_unaligned_overlap()
    test_discrete_unaligned_reorder()
    test_discrete_subset_with_bg()
    test_continuous_continuous()
    test_hybrid_composition()