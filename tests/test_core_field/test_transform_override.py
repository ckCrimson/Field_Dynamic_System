import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.core.field.mappings import ContinuousFieldMapper
from src.field_dynamic_system.core.field.transform import NonLinearTransform
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransformer
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace


def test_override_background_transform():
    print("\n=== TEST: Transform with Background Override ===")

    # 1. Setup Space & Algebra
    # We use a mock space just to hold keys
    space = AbstractDiscreteStateSpace(["explicit_point", "background_point"])
    algebra = RealFieldAlgebra()

    # 2. Define Old Background: f(x) = 10.0 (Constant)
    def old_bg(state):
        # Return 10.0 for everything
        N = state.shape[0] if hasattr(state, 'shape') else 1
        return jnp.full((N, 1), 10.0)

    # 3. Create Mapper
    mapper = ContinuousFieldMapper(space, algebra, bg_func=old_bg)

    # 4. Set an Explicit Value (The "Sparse Cache")
    # State "explicit_point" = 2.0
    mapper.set_value_at("explicit_point", 2.0)

    # ---------------------------------------------------------
    # SCENE A: Standard Transform (Composition)
    # Transform: T(x) = x + 1
    # Expected:
    #   Explicit: 2.0 + 1 = 3.0
    #   Background: 10.0 + 1 = 11.0
    # ---------------------------------------------------------
    print("  -> Scene A: Standard Composition (No Override)")
    op_add = NonLinearTransform(lambda x: x + 1.0)

    res_A = FieldSpaceTransformer.apply(mapper, op_add,RealFieldAlgebra())

    val_explicit_A = res_A.get_fields_at("explicit_point")[0].value
    val_bg_A = res_A.get_fields_at("background_point")[0].value

    assert jnp.allclose(val_explicit_A, 3.0)
    assert jnp.allclose(val_bg_A, 11.0)
    print("     ✅ Composition Correct.")

    # ---------------------------------------------------------
    # SCENE B: Transform with Override
    # Transform: T(x) = x + 1
    # Override BG: g(x) = -99.0
    # Expected:
    #   Explicit: 2.0 + 1 = 3.0 (Transform STILL applies to cache)
    #   Background: -99.0 (Override replaces old bg completely)
    # ---------------------------------------------------------
    print("  -> Scene B: With Override Function")

    def override_bg(state):
        N = state.shape[0] if hasattr(state, 'shape') else 1
        return jnp.full((N, 1), -99.0)

    res_B = FieldSpaceTransformer.apply(mapper, op_add, RealFieldAlgebra(),override_bg_func=override_bg)

    val_explicit_B = res_B.get_fields_at("explicit_point")[0].value
    val_bg_B = res_B.get_fields_at("background_point")[0].value

    # Check 1: Did explicit value still get transformed?
    assert jnp.allclose(val_explicit_B, 3.0)

    # Check 2: Did background get overridden (ignoring the +1 transform)?
    assert jnp.allclose(val_bg_B, -99.0)

    print("     ✅ Override Correct (Cache transformed, BG replaced).")


if __name__ == "__main__":
    test_override_background_transform()