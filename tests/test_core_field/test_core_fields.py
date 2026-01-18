import time
import jax.numpy as jnp
import pytest
from jax import config

# Enable 64-bit precision
config.update("jax_enable_x64", True)

# --- Imports ---
from src.field_dynamic_system.core.field.mappings import FieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra, ComplexFieldAlgebra
from src.field_dynamic_system.core.field.data import RealFieldValue, extract_val
from src.field_dynamic_system.core.state.interfaces import IDiscreteStateSpace


# ==========================================
# 0. Fixed Mock State Spaces
# ==========================================

class MockIntegerSpace(IDiscreteStateSpace):
    """
    A simple 1D line of integer states: 0, 1, ..., N-1.
    Implements ALL abstract methods required by IDiscreteStateSpace.
    """

    def __init__(self, n_states=100):
        self._n_states = n_states

    # --- Interface Properties ---
    @property
    def n_states(self):
        # Used by FieldMapper logic
        return self._n_states

    @property
    def num_states(self):
        # Required by IDiscreteStateSpace Interface
        return self._n_states

    @property
    def states(self):
        # Required by IDiscreteStateSpace Interface
        return jnp.arange(self._n_states)

    @property
    def capacity(self):
        return self._n_states

    # --- Core Lookup Methods ---
    def get_index_of(self, state_obj):
        # Required by IDiscreteStateSpace Interface
        # Validates and returns index
        if not (0 <= state_obj < self._n_states):
            raise ValueError("State out of bounds")
        return int(state_obj)

    def state_to_index(self, state_obj):
        # Compatibility Alias for FieldMapper if it uses this name
        return self.get_index_of(state_obj)

    def create_subset(self, indices):
        # Required by Interface (Dummy implementation for tests)
        return MockIntegerSpace(len(indices))

    # --- JAX Pytree Hooks (Required) ---
    def _tree_flatten(self):
        # Flatten for JIT: (Children, Metadata)
        return ((), {"n_states": self._n_states})

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(n_states=aux["n_states"])


class MockGridSpace(IDiscreteStateSpace):
    """
    A 2D Grid: (x, y). Maps (x,y) -> Index.
    """

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self._n_states = width * height

    # --- Interface Properties ---
    @property
    def n_states(self): return self._n_states

    @property
    def num_states(self): return self._n_states

    @property
    def states(self): return jnp.arange(self._n_states)  # Simplified return

    @property
    def capacity(self): return self._n_states

    # --- Core Lookup Methods ---
    def get_index_of(self, state_obj):
        # Expecting tuple (x, y)
        x, y = state_obj
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("Grid State out of bounds")
        return int(y * self.width + x)

    def state_to_index(self, state_obj):
        return self.get_index_of(state_obj)

    def create_subset(self, indices):
        return MockGridSpace(1, len(indices))

    # --- JAX Pytree Hooks ---
    def _tree_flatten(self):
        return ((), {"width": self.width, "height": self.height})

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(width=aux["width"], height=aux["height"])


# ==========================================
# TEST 1: Integer Space + Real Field
# ==========================================

def test_integer_real_field_methods():
    print("\n=== TEST 1: Integer Space (N=100) | Real Field ===")

    # Setup
    space = MockIntegerSpace(n_states=100)
    algebra = RealFieldAlgebra()

    # 1. Test Default (Zero Field)
    mapper_zero = FieldMapper(space, algebra)
    all_indices = jnp.arange(space.n_states)
    values = mapper_zero.get_fields_at(all_indices)

    assert values.shape == (100, 1)
    assert jnp.all(values == 0.0)
    print("✅ Default Zero Field: Verified")

    # 2. Test Define Constant (0.5)
    mapper_const = FieldMapper.define_constant_field(space, algebra, 0.5)
    values = mapper_const.get_fields_at(all_indices)
    assert jnp.all(values == 0.5)
    print("✅ Define Constant (0.5): Verified")

    # 3. Test Define Unity
    mapper_unity = FieldMapper.define_unity_field(space, algebra)
    values = mapper_unity.get_fields_at(all_indices)
    assert jnp.all(values == 1.0)
    print("✅ Define Unity: Verified")

    # 4. Test Impulse (Hybrid Logic)
    target_state = 42
    mapper_impulse = FieldMapper.define_impulse_field(space, algebra, target_state)
    values = mapper_impulse.get_fields_at(all_indices)

    assert values[target_state] == 1.0, "Impulse target failed"
    assert jnp.sum(values) == 1.0, "Impulse conservation failed"
    print("✅ Define Impulse: Verified")

    # 5. Test Manual Overrides
    mapper_zero.set_value_at(state_obj=10, value=5.0)
    vals = mapper_zero.get_fields_at(jnp.array([10, 11]))
    assert vals[0] == 5.0
    assert vals[1] == 0.0
    print("✅ Manual Overrides: Verified")


# ==========================================
# TEST 2: Performance
# ==========================================

def test_grid_performance():
    print("\n=== TEST 2: 2D Grid (100x100 = 10k States) | Performance ===")
    W, H = 100, 100
    space = MockGridSpace(W, H)
    algebra = RealFieldAlgebra()

    center_state = (50, 50)
    mapper = FieldMapper.define_impulse_field(space, algebra, center_state)
    all_indices = jnp.arange(space.n_states)

    # Warmup
    _ = mapper.get_fields_at(all_indices).block_until_ready()

    start = time.time()
    for _ in range(100):  # 100 loops
        _ = mapper.get_fields_at(all_indices).block_until_ready()

    avg = (time.time() - start) / 100
    print(f"✅ Performance: {avg * 1000:.4f} ms per 10k state fetch")


# ==========================================
# TEST 3: Complex Field Checks
# ==========================================

def test_complex_field_basics():
    print("\n=== TEST 3: Complex Field Checks ===")
    space = MockIntegerSpace(10)
    algebra = ComplexFieldAlgebra()

    # Unity Check
    mapper = FieldMapper.define_unity_field(space, algebra)
    val = mapper.get_fields_at(jnp.array([0]))

    assert jnp.iscomplexobj(val)
    assert val[0] == 1.0 + 0.0j
    print("✅ Complex Unity: Verified")


def test_complex_field_logic():
    print("\n=== TEST 4: Complex Logic (Norm) ===")
    space = MockIntegerSpace(10)
    algebra = ComplexFieldAlgebra()

    # State i (0 + 1j)
    val_i = 0.0 + 1.0j
    mapper_i = FieldMapper.define_constant_field(space, algebra, val_i)

    fields = mapper_i.get_fields_at(jnp.array([0]))
    norms = algebra.norm(fields)

    norm_val = extract_val(norms)
    assert jnp.allclose(norm_val, 1.0)
    print("✅ Complex Norm (|i|=1): Verified")