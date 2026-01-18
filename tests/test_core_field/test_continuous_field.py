import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.mappings import FieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.core.state import *
from src.field_dynamic_system.core.field.data import extract_val


# ==========================================
# 1. MOCK SPACES
# ==========================================

# --- A. Discrete Space ---
class MockHybridIntegerSpace(IDiscreteStateSpace):
    def __init__(self, n=10): self._n = n

    @property
    def n_states(self): return self._n

    @property
    def num_states(self): return self._n

    @property
    def states(self): return jnp.arange(self._n)

    @property
    def capacity(self): return self._n

    def get_index_of(self, s): return int(s)

    def state_to_index(self, s): return int(s)

    def create_subset(self, idx): return self

    # JAX HOOKS
    def tree_flatten(self): return ((), {"n": self._n})

    @classmethod
    def tree_unflatten(cls, aux, _): return cls(n=aux["n"])

    # INTERFACE HOOKS (Satisfy abstract class)
    def _tree_flatten(self): return self.tree_flatten()

    @classmethod
    def _tree_unflatten(cls, aux, children): return cls.tree_unflatten(aux, children)


# --- B. Continuous Space ---
class MockHybridContinuousSpace(StateSpace):
    def tree_flatten(self): return ((), None)

    @classmethod
    def tree_unflatten(cls, aux, children): return cls()

    # INTERFACE HOOKS
    def _tree_flatten(self): return self.tree_flatten()

    @classmethod
    def _tree_unflatten(cls, aux, children): return cls.tree_unflatten(aux, children)


# --- C. Hashable Vector State ---
class HybridPoint2D(VectorState):
    def __init__(self, x, y):
        object.__setattr__(self, 'x', float(x))
        object.__setattr__(self, 'y', float(y))
        val = jnp.array([x, y])
        object.__setattr__(self, '_val', val)

    @property
    def value(self): return self._val

    def __hash__(self): return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, HybridPoint2D) and self.x == other.x and self.y == other.y

    def __repr__(self): return f"P({self.x}, {self.y})"


# ==========================================
# 2. SAFE JAX REGISTRATION
# ==========================================
def safe_register(cls):
    try:
        register_pytree_node_class(cls)
    except ValueError:
        pass


safe_register(MockHybridIntegerSpace)
safe_register(MockHybridContinuousSpace)


# ==========================================
# 3. TESTS
# ==========================================

def test_discrete_set_get():
    print("\n=== TEST 1: Discrete Space (JAX Buffer) ===")
    space = MockHybridIntegerSpace(10)
    algebra = RealFieldAlgebra()

    mapper = FieldMapper(space, algebra)

    # SET
    mapper.set_value_at(5, 99.0)

    # GET
    val_5 = mapper.get_fields_at(5)
    assert val_5.value == 99.0

    val_0 = mapper.get_fields_at(0)
    assert val_0.value == 0.0

    # Batch
    batch_vals = mapper.get_fields_at(jnp.array([4, 5, 6]))
    assert jnp.allclose(batch_vals[1], 99.0)

    print("✅ Discrete Logic Verified")


def test_continuous_set_get():
    print("\n=== TEST 2: Continuous Space (Sparse Cache) ===")
    space = MockHybridContinuousSpace()
    algebra = RealFieldAlgebra()

    def bg_func(states):
        return jnp.sum(states, axis=1, keepdims=True)

    mapper = FieldMapper(space, algebra, background_func=bg_func)

    p1 = HybridPoint2D(2.0, 3.0)
    p2 = HybridPoint2D(10.0, 10.0)

    # GET (Background Calculation)
    val_p1 = mapper.get_fields_at(p1)
    assert val_p1.value == 5.0

    # SET (Override)
    mapper.set_value_at(p2, 100.0)

    # GET (Override)
    val_p2 = mapper.get_fields_at(p2)
    assert val_p2.value == 100.0

    # Verify Cache
    assert p1 in mapper.sparse_cache
    assert p2 in mapper.sparse_cache

    print("✅ Continuous Logic Verified")


if __name__ == "__main__":
    test_discrete_set_get()
    test_continuous_set_get()