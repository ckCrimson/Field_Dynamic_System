from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Union, Dict
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.tree_util import register_pytree_node_class

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.interfaces import StateSpace, IDiscreteStateSpace
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.core.field.data import extract_val, FieldValue


# ... [Interfaces and Classes DiscreteFieldMapper / ContinuousFieldMapper remain EXACTLY the same as before] ...
# ... [Just copy IFieldMapper, DiscreteFieldMapper, ContinuousFieldMapper from the previous successful file] ...

# PASTE THE CLASSES HERE (IFieldMapper, DiscreteFieldMapper, ContinuousFieldMapper)
# To save space, I will focus on the FACTORY section which needs the change.

# ==========================================
# 0. THE INTERFACE (Same)
# ==========================================
class IFieldMapper(ABC):
    state_space: StateSpace
    algebra: IFieldAlgebra
    background_func: Optional[Callable]

    @abstractmethod
    def get_fields_at(self, query: Any) -> Union[jnp.ndarray, FieldValue]: pass

    @abstractmethod
    def set_value_at(self, state_obj: Any, value: Any): pass


# ==========================================
# 1. DISCRETE IMPLEMENTATION (Same)
# ==========================================
@register_pytree_node_class
class DiscreteFieldMapper(IFieldMapper):
    def __init__(self, state_space: IDiscreteStateSpace, algebra: IFieldAlgebra, bg_func=None):
        self.state_space = state_space
        self.algebra = algebra
        self.dim = getattr(algebra, 'dim', 1)
        self.dtype = getattr(algebra, 'dtype', jnp.float64)

        # If bg_func is None, we rely on buffer.
        # If provided, we wrap it.
        if bg_func is None:
            self.background_func = None
        else:
            self.background_func = lambda s: extract_val(bg_func(s))

        n = getattr(state_space, 'n_states', 1000)
        self.explicit_buffer = jnp.zeros((n, self.dim), dtype=self.dtype)
        self.mask_buffer = jnp.zeros((n, 1), dtype=bool)

    def tree_flatten(self):
        return ((self.explicit_buffer, self.mask_buffer), (self.state_space, self.algebra, self.background_func))

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.state_space, obj.algebra, obj.background_func = aux
        obj.explicit_buffer, obj.mask_buffer = children
        obj.dim, obj.dtype = getattr(obj.algebra, 'dim', 1), getattr(obj.algebra, 'dtype', jnp.float64)
        return obj

    @jit
    def _get_batch(self, indices):
        # 1. Background
        if self.background_func is None:
            bg_vals = jnp.zeros((indices.shape[0], self.dim), dtype=self.dtype)
        else:
            bg_vals = self.background_func(indices)

        # 2. Buffer
        idx_safe = indices.astype(jnp.int32)
        explicit_vals = self.explicit_buffer[idx_safe]
        masks = self.mask_buffer[idx_safe]
        return jnp.where(masks, explicit_vals, bg_vals)

    def get_fields_at(self, query: Union[int, jnp.ndarray]) -> Union[FieldValue, jnp.ndarray]:
        if isinstance(query, jnp.ndarray): return self._get_batch(query)
        if isinstance(query, (int, float)): return FieldValue(self._get_batch(jnp.array([query], dtype=jnp.int32))[0])
        return self._get_batch(jnp.asarray(query, dtype=jnp.int32))

    def set_value_at(self, state_obj: Any, value: Any):
        idx = self.state_space.get_index_of(state_obj)
        raw_val = jnp.array(extract_val(value), dtype=self.dtype)
        if raw_val.ndim == 0: raw_val = raw_val.reshape(1, 1)
        if self.dim > 1:
            raw_val = jnp.broadcast_to(raw_val, (1, self.dim))
        elif raw_val.ndim == 1 and raw_val.shape[0] == self.dim:
            raw_val = raw_val.reshape(1, self.dim)

        idx_arr = jnp.array([idx])
        self.explicit_buffer = self.explicit_buffer.at[idx_arr].set(raw_val)
        self.mask_buffer = self.mask_buffer.at[idx_arr].set(True)
        return self


# ==========================================
# 2. CONTINUOUS IMPLEMENTATION (Same)
# ==========================================
@register_pytree_node_class
class ContinuousFieldMapper(IFieldMapper):
    def __init__(self, state_space: StateSpace, algebra: IFieldAlgebra, bg_func=None):
        self.state_space = state_space;
        self.algebra = algebra
        self.dim = getattr(algebra, 'dim', 1);
        self.dtype = getattr(algebra, 'dtype', jnp.float64)
        self.background_func = (lambda s: jnp.zeros((s.shape[0], self.dim), dtype=self.dtype)) if bg_func is None else (
            lambda s: extract_val(bg_func(s)))
        self.sparse_cache: Dict[Any, jnp.ndarray] = {};
        self.explicit_buffer = None

    def get_fields_at(self, query: Any) -> Union[FieldValue, jnp.ndarray]:
        is_batch = isinstance(query, (list, tuple, jnp.ndarray)) and not isinstance(query, VectorState)
        if not is_batch:
            state = query
            if state in self.sparse_cache: return FieldValue(self.sparse_cache[state])
            inp_arr = jnp.atleast_2d(jnp.asarray(state.value if hasattr(state, 'value') else state))
            val = self.background_func(inp_arr)[0]
            self.sparse_cache[state] = val
            return FieldValue(val)

        results = []
        for state in query:
            if state in self.sparse_cache:
                results.append(self.sparse_cache[state])
            else:
                inp = jnp.atleast_2d(jnp.asarray(state.value if hasattr(state, 'value') else state))
                val = self.background_func(inp)[0]
                self.sparse_cache[state] = val
                results.append(val)
        return jnp.stack(results)

    def set_value_at(self, state_obj: Any, value: Any):
        self.sparse_cache[state_obj] = jnp.array(extract_val(value), dtype=self.dtype)
        return self

    def tree_flatten(self):
        return ((), (self.state_space, self.algebra, self.background_func, self.sparse_cache))

    @classmethod
    def tree_unflatten(cls, aux, _):
        obj = cls.__new__(cls);
        obj.state_space, obj.algebra, obj.background_func, obj.sparse_cache = aux
        obj.explicit_buffer = None;
        obj.dim, obj.dtype = getattr(obj.algebra, 'dim', 1), getattr(obj.algebra, 'dtype', jnp.float64)
        return obj


# ==========================================
# 3. FACTORY (UPDATED FOR EAGER MATERIALIZATION)
# ==========================================

class FieldMapper:
    def __new__(cls, state_space, field_algebra, background_func=None):
        if isinstance(state_space, IDiscreteStateSpace):
            return DiscreteFieldMapper(state_space, field_algebra, background_func)
        else:
            return ContinuousFieldMapper(state_space, field_algebra, background_func)

    @staticmethod
    def define_constant_field(state_space, field_algebra, value):
        raw_val = extract_val(value)
        dim = getattr(field_algebra, 'dim', 1)
        dtype = getattr(field_algebra, 'dtype', jnp.float64)

        # Pre-calculate value shape
        val_shaped = jnp.array(raw_val, dtype=dtype)
        if val_shaped.ndim == 0: val_shaped = val_shaped.reshape(1, 1)
        if dim > 1 and val_shaped.shape[1] == 1: val_shaped = jnp.broadcast_to(val_shaped, (1, dim))

        # OPTIMIZATION: If Discrete, fill buffer NOW. Avoid closure.
        if isinstance(state_space, IDiscreteStateSpace):
            mapper = DiscreteFieldMapper(state_space, field_algebra, bg_func=None)

            # Create full constant array
            n_states = state_space.num_states
            full_buffer = jnp.broadcast_to(val_shaped, (n_states, dim))

            # Inject into buffer
            mapper.explicit_buffer = full_buffer
            # Mark all as overridden (True) so we ignore the (None) background
            mapper.mask_buffer = jnp.ones((n_states, 1), dtype=bool)
            return mapper

        # Continuous Fallback (Lazy)
        def const_func(s):
            s = jnp.asarray(s);
            N = s.shape[0] if s.ndim > 0 else 1
            return jnp.broadcast_to(val_shaped, (N, dim))

        return ContinuousFieldMapper(state_space, field_algebra, background_func=const_func)

    @staticmethod
    def define_unity_field(state_space, field_algebra):
        # OPTIMIZATION: If Discrete, fill buffer NOW.
        if isinstance(state_space, IDiscreteStateSpace):
            mapper = DiscreteFieldMapper(state_space, field_algebra, bg_func=None)
            n_states = state_space.n_states
            # Get unity scalar/vector
            unity = extract_val(field_algebra.get_unity(shape=(1,)))
            full_buffer = jnp.broadcast_to(unity, (n_states, mapper.dim))

            mapper.explicit_buffer = full_buffer
            mapper.mask_buffer = jnp.ones((n_states, 1), dtype=bool)
            return mapper

        def unity_func(s):
            s = jnp.asarray(s);
            N = s.shape[0] if s.ndim > 0 else 1
            return extract_val(field_algebra.get_unity(shape=(N,)))

        return ContinuousFieldMapper(state_space, field_algebra, background_func=unity_func)

    @staticmethod
    def define_impulse_field(state_space, field_algebra, target_state):
        # This one is already "Collapsed" by default (starts with None bg, sets one value)
        mapper = FieldMapper(state_space, field_algebra, background_func=None)
        unity = field_algebra.get_unity(shape=(1,))
        mapper.set_value_at(target_state, unity)
        return mapper