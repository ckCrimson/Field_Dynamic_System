from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Union, Dict, List, Sequence
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import register_pytree_node_class

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.interfaces import StateSpace, IDiscreteStateSpace, State
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.core.field.data import extract_val, FieldValue


# =========================================================
# 1. THE INTERFACE
# =========================================================
class IFieldMapper(ABC):
    state_space: StateSpace
    algebra: IFieldAlgebra
    background_func: Optional[Callable]

    @abstractmethod
    def get_fields_at(self, query: Any) -> Any: pass

    @abstractmethod
    def set_value_at(self, state_obj: Any, value: Any): pass

    @property
    @abstractmethod
    def algebra_type(self): pass


# =========================================================
# 2. DISCRETE ENGINE
# =========================================================
@register_pytree_node_class
class DiscreteFieldMapper(IFieldMapper):
    def __init__(self, state_space: IDiscreteStateSpace, algebra: IFieldAlgebra,
                 explicit_buffer=None, mask_buffer=None, bg_func=None):
        self.state_space = state_space
        self.algebra = algebra
        self.dim = getattr(algebra, 'dim', 1)
        self.dtype = getattr(algebra, 'dtype', jnp.float64)
        self.background_func = bg_func

        if explicit_buffer is not None:
            self.explicit_buffer = explicit_buffer
            self.mask_buffer = mask_buffer if mask_buffer is not None else jnp.ones((explicit_buffer.shape[0], 1),
                                                                                    dtype=bool)
        else:
            n = max(state_space.num_states, 1)
            self.explicit_buffer = jnp.zeros((n, self.dim), dtype=self.dtype)
            self.mask_buffer = jnp.zeros((n, 1), dtype=bool)

    # --- FACTORIES ---
    @classmethod
    def constant(cls, state_space, algebra, value):
        raw_val = extract_val(value)
        dim, dtype = getattr(algebra, 'dim', 1), getattr(algebra, 'dtype', jnp.float64)
        val_shaped = jnp.array(raw_val, dtype=dtype)
        if val_shaped.ndim == 0: val_shaped = val_shaped.reshape(1, 1)
        if dim > 1 and val_shaped.shape[1] == 1: val_shaped = jnp.broadcast_to(val_shaped, (1, dim))

        n_states = max(state_space.num_states, 1)
        full_buffer = jnp.broadcast_to(val_shaped, (n_states, dim))

        return cls(state_space, algebra, explicit_buffer=full_buffer)

    @classmethod
    def unity(cls, state_space, algebra):
        unity_val = extract_val(algebra.get_unity(shape=(1,)))
        return cls.constant(state_space, algebra, unity_val)

    @classmethod
    def impulse(cls, state_space, algebra, target_state):
        mapper = cls(state_space, algebra)
        unity = algebra.get_unity(shape=(1,))
        mapper.set_value_at(target_state, unity)
        return mapper

    # --- BATCH ---
    @classmethod
    def batch_compose(cls, mappers: Sequence['DiscreteFieldMapper'], op_name="__add__"):
        if not mappers: return None
        first = mappers[0]
        stack_vals = jnp.stack([m.explicit_buffer for m in mappers])
        stack_mask = jnp.stack([m.mask_buffer for m in mappers])

        if op_name == "__add__":
            new_buffer = jnp.sum(stack_vals, axis=0)
        elif op_name == "__mul__":
            new_buffer = jnp.prod(stack_vals, axis=0)
        else:
            new_buffer = stack_vals[0]
            for i in range(1, len(stack_vals)):
                new_buffer = new_buffer + stack_vals[i]

        new_mask = jnp.any(stack_mask, axis=0)
        return cls(first.state_space, first.algebra, explicit_buffer=new_buffer, mask_buffer=new_mask)

    # --- CORE LOGIC ---
    def _ensure_capacity(self, required_idx):
        current_size = self.explicit_buffer.shape[0]
        if required_idx < current_size: return

        new_size = self.state_space.num_states
        diff = new_size - current_size
        if diff > 0:
            val_pad = jnp.zeros((diff, self.dim), dtype=self.dtype)
            mask_pad = jnp.zeros((diff, 1), dtype=bool)
            self.explicit_buffer = jnp.vstack([self.explicit_buffer, val_pad])
            self.mask_buffer = jnp.vstack([self.mask_buffer, mask_pad])

    def get_fields_at(self, query):
        if isinstance(query, (list, tuple)):
            indices = self.state_space.register_states(query)
        elif isinstance(query, (int, jnp.ndarray, np.ndarray, float)):
            indices = jnp.asarray(query, dtype=jnp.int32)
        else:
            indices = self.state_space.register_states([query])

        raw = self._get_batch(indices)
        return [FieldValue(row) for row in raw]

    @jit
    def _get_batch(self, indices):
        valid_mask = (indices >= 0) & (indices < self.explicit_buffer.shape[0])
        safe_indices = jnp.where(valid_mask, indices, 0)
        return jnp.where(valid_mask.reshape(-1, 1),
                         self.explicit_buffer[safe_indices],
                         jnp.zeros((1, self.dim), dtype=self.dtype))

    def set_value_at(self, state_obj, value):
        idx_arr = self.state_space.register_states([state_obj])
        idx = int(idx_arr[0])
        self._ensure_capacity(idx)

        raw_val = jnp.array(extract_val(value), dtype=self.dtype).ravel()
        self.explicit_buffer = self.explicit_buffer.at[idx].set(raw_val)
        self.mask_buffer = self.mask_buffer.at[idx].set(True)
        return self

    @property
    def algebra_type(self):
        return type(self.algebra)

    def tree_flatten(self):
        return ((self.explicit_buffer, self.mask_buffer), (self.state_space, self.algebra, self.background_func))

    @classmethod
    def tree_unflatten(cls, aux, children):
        state_space, algebra, bg_func = aux
        return cls(state_space, algebra, explicit_buffer=children[0], mask_buffer=children[1], bg_func=bg_func)


# =========================================================
# 3. CONTINUOUS ENGINE
# =========================================================
@register_pytree_node_class
class ContinuousFieldMapper(IFieldMapper):
    def __init__(self, state_space: StateSpace, algebra: IFieldAlgebra, bg_func=None, sparse_cache=None):
        self.state_space = state_space
        self.algebra = algebra
        self.dim = getattr(algebra, 'dim', 1)
        self.dtype = getattr(algebra, 'dtype', jnp.float64)

        self.background_func = bg_func if bg_func else (
            lambda s: jnp.zeros((len(s) if hasattr(s, '__len__') else 1, self.dim), dtype=self.dtype))
        self.sparse_cache = sparse_cache if sparse_cache is not None else {}

    # --- FACTORIES ---
    @classmethod
    def constant(cls, state_space, algebra, value):
        raw_val = extract_val(value)
        dim = getattr(algebra, 'dim', 1)

        def const_func(s):
            s_arr = jnp.asarray(s)
            N = s_arr.shape[0] if s_arr.ndim > 0 else 1
            return jnp.broadcast_to(raw_val, (N, dim))

        return cls(state_space, algebra, bg_func=const_func)

    @classmethod
    def unity(cls, state_space, algebra):
        val = extract_val(algebra.get_unity(shape=(1,)))
        return cls.constant(state_space, algebra, val)

    @classmethod
    def impulse(cls, state_space, algebra, target_state):
        mapper = cls(state_space, algebra)
        unity = algebra.get_unity(shape=(1,))
        mapper.set_value_at(target_state, unity)
        return mapper

    # --- BATCH ---
    @classmethod
    def batch_compose(cls, mappers: Sequence['ContinuousFieldMapper'], op_name="__add__"):
        if not mappers: return None
        first = mappers[0]

        def batch_bg(state):
            results = [m.background_func(state) for m in mappers]
            stack_res = jnp.stack(results)
            if op_name == "__add__":
                return jnp.sum(stack_res, axis=0)
            elif op_name == "__mul__":
                return jnp.prod(stack_res, axis=0)
            return results[0]

        new_cache = {}
        all_keys = set().union(*[m.sparse_cache.keys() for m in mappers])

        for s in all_keys:
            vals = [m.get_fields_at(s)[0].value for m in mappers]
            stack_vals = jnp.stack(vals)
            if op_name == "__add__":
                res = jnp.sum(stack_vals, axis=0)
            elif op_name == "__mul__":
                res = jnp.prod(stack_vals, axis=0)
            else:
                res = stack_vals[0]
            new_cache[s] = res

        return cls(first.state_space, first.algebra, bg_func=batch_bg, sparse_cache=new_cache)

    # --- CORE LOGIC ---
    def get_fields_at(self, query):
        targets = query if isinstance(query, (list, tuple)) else [query]
        results = []
        for state in targets:
            # 1. Check Sparse Cache (Dictionary)
            if state in self.sparse_cache:
                raw = self.sparse_cache[state]
            else:
                # 2. Compute Background
                inp = state.values if hasattr(state, 'values') else state

                try:
                    # OPTIMISTIC: Try to convert to JAX array (Vector/Coord)
                    inp_arr = jnp.atleast_2d(jnp.array(inp))
                    raw = self.background_func(inp_arr)[0]
                except (TypeError, ValueError):
                    # FALLBACK: If state is a String/Object, pass raw to function
                    # The function must be able to handle this type.
                    raw = self.background_func(inp)[0]

            results.append(FieldValue(raw))
        return results

    def set_value_at(self, state, value):
        if hasattr(self.state_space, 'register_states'):
            self.state_space.register_states([state])

        self.sparse_cache[state] = jnp.array(extract_val(value), dtype=self.dtype)
        return self

    @property
    def algebra_type(self):
        return type(self.algebra)

    def tree_flatten(self):
        return ((), (self.state_space, self.algebra, self.background_func, self.sparse_cache))

    @classmethod
    def tree_unflatten(cls, aux, _):
        return cls(*aux)


# =========================================================
# 4. THE FACADE / FACTORY
# =========================================================
class FieldMapper:
    def __new__(cls, state_space, field_algebra, background_func=None):
        if isinstance(state_space, IDiscreteStateSpace):
            return DiscreteFieldMapper(state_space, field_algebra, bg_func=background_func)
        return ContinuousFieldMapper(state_space, field_algebra, bg_func=background_func)

    @staticmethod
    def define_constant_field(state_space, field_algebra, value):
        if isinstance(state_space, IDiscreteStateSpace):
            return DiscreteFieldMapper.constant(state_space, field_algebra, value)
        return ContinuousFieldMapper.constant(state_space, field_algebra, value)

    @staticmethod
    def define_unity_field(state_space, field_algebra):
        if isinstance(state_space, IDiscreteStateSpace):
            return DiscreteFieldMapper.unity(state_space, field_algebra)
        return ContinuousFieldMapper.unity(state_space, field_algebra)

    @staticmethod
    def define_impulse_field(state_space, field_algebra, target_state):
        if isinstance(state_space, IDiscreteStateSpace):
            return DiscreteFieldMapper.impulse(state_space, field_algebra, target_state)
        return ContinuousFieldMapper.impulse(state_space, field_algebra, target_state)