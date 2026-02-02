from dataclasses import dataclass
from collections import defaultdict
from typing import Set, List, Union, Sequence, Any, Optional, Callable

import jax.numpy as jnp
import jax
import numpy as np

# Ensure these imports match your project structure
from .interfaces import IDiscreteStateSpace, StateEncoder, StateSpace, State
from .encoding import BitMaskingEncoding, VectorEncoding
from .state import VectorState


# --- 1. THE LAZY PROXY ---
class LazyStateProxy:
    """
    Universal Proxy: Wraps any raw collection (Numpy/List) and creates
    objects only when accessed. Exposed .raw_data for Topology speed.
    """

    def __init__(self, raw_collection: Sequence[Any], wrapper_func: Callable[[Any], Any]):
        self.raw_data = raw_collection
        self._wrapper = wrapper_func

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self._wrapper(self.raw_data[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# --- 2. ABSTRACT BASE (FIXED) ---
class AbstractDiscreteStateSpace(IDiscreteStateSpace):
    """
    Dynamic State Space.
    Maintains a bijection between State Objects and Integer Indices.
    """

    def __init__(self,
                 states: Union[Set[Any], Sequence[Any]],
                 encoder: Optional[StateEncoder] = None):

        # 1. Internal Storage
        initial_list = sorted(list(set(states)), key=lambda x: str(x))

        self._idx_to_state = initial_list
        self._state_to_idx = {s: i for i, s in enumerate(initial_list)}

        # 2. Encoder
        self._encoder = encoder if encoder else BitMaskingEncoding(initial_list)

        # 3. Post-Init Hook
        self._post_init()

    def _post_init(self):
        pass

    # [THE FACTORY]
    @classmethod
    def from_raw_data(cls, raw_data: Sequence[Any], wrapper: Callable[[Any], Any], **kwargs):
        obj = cls.__new__(cls)
        obj._idx_to_state = LazyStateProxy(raw_data, wrapper)
        obj._state_to_idx = {}  # Empty for speed
        obj._encoder = kwargs.get('encoder')
        obj._raw_init_hook(raw_data, **kwargs)
        return obj

    def _raw_init_hook(self, raw_data, **kwargs):
        pass

    # --- Standard Properties ---

    @property
    def num_states(self) -> int:
        return len(self._idx_to_state)

    @property
    def states(self) -> List[Any]:
        return self._idx_to_state

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    # --- CRITICAL FIX: CONTAINS METHOD ---
    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """
        Robust Membership Check.
        """
        # Case A: JAX Array (Indices)
        if isinstance(state, (jnp.ndarray, np.ndarray)) and (jnp.issubdtype(state.dtype, jnp.integer)):
            return (state >= 0) & (state < self.num_states)

        # Case B: Python List (Batch of Objects)
        if isinstance(state, (list, tuple)):
            # Check against dictionary keys O(1)
            mask = [s in self._state_to_idx for s in state]
            return jnp.array(mask, dtype=bool)

        # Case C: Single Object
        return state in self._state_to_idx

    def get_index_of(self, state_obj: Any) -> int:
        return self._state_to_idx.get(state_obj, -1)

    # --- Map Operation ---

        # --- MISSING METHOD ADDED HERE ---
    def get_matrix(self):
        """
        Converts states to a numerical Matrix (N, D).
        Required by Kernels (Gaussian, etc.) to compute distances.
        """
        if not self._idx_to_state:
            return np.empty((0, 0), dtype=np.float32)

        first = self._idx_to_state[0]

        # Case 1: Simple Scalars (Integers/Floats) -> Convert to (N, 1) column vector
        # This handles your test case: [-5, -4, ... 5]
        if isinstance(first, (int, float, np.number)):
            return np.array(self._idx_to_state, dtype=np.float32).reshape(-1, 1)

        # Case 2: Vectors/Tuples -> Convert to (N, D) matrix
        if hasattr(first, '__len__') and not isinstance(first, str):
            try:
                return np.array([list(s) for s in self._idx_to_state], dtype=np.float32)
            except:
                # Handle VectorState objects
                if hasattr(first, 'values'):
                    return np.array([s.values for s in self._idx_to_state], dtype=np.float32)
                elif hasattr(first, 'coordinates'):
                    return np.array([s.coordinates for s in self._idx_to_state], dtype=np.float32)

        # Case 3: Objects without coordinates -> Return empty (Kernel will fail gracefully or warn)
        return np.zeros((len(self._idx_to_state), 0), dtype=np.float32)

    def map(self, func: Callable[[Any], Any]) -> Union[jnp.ndarray, List[Any], np.ndarray]:
        """
        Base Map implementation. Safe for all data types.
        """
        results = [func(s) for s in self.states]

        if not results: return jnp.array([])

        first_elem = results[0]
        if isinstance(first_elem, (str, VectorState, dict, set)):
            return results

        try:
            return jnp.stack(results)
        except (TypeError, ValueError):
            return results

    # --- Set Ops ---
    def create_subset(self, states: List[Any]) -> 'AbstractDiscreteStateSpace':
        return self.__class__(states)

    def union(self, other: StateSpace) -> StateSpace:
        if isinstance(other, AbstractDiscreteStateSpace):
            combined = list(set(self.states) | set(other.states))
            return self.create_subset(combined)
        raise TypeError(f"Cannot union with {type(other)}")

    def intersection(self, other: StateSpace) -> StateSpace:
        if isinstance(other, AbstractDiscreteStateSpace):
            common = list(set(self.states) & set(other.states))
            return self.create_subset(common)
        raise TypeError(f"Cannot intersect with {type(other)}")

    # --- Register (Keep for compatibility) ---
    def register_states(self, states_batch: Sequence[Any]) -> jnp.ndarray:
        indices = []
        for s in states_batch:
            if s not in self._state_to_idx:
                new_idx = len(self._idx_to_state)
                self._state_to_idx[s] = new_idx
                self._idx_to_state.append(s)
                self._on_state_added(s)
            indices.append(self._state_to_idx[s])
        return jnp.array(indices, dtype=jnp.int32)

    def _on_state_added(self, state):
        pass

    # --- Pytree ---
    def _tree_flatten(self):
        return (), (self._idx_to_state, self._encoder)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        states, encoder = aux
        obj = cls.__new__(cls)
        obj._idx_to_state = states
        obj._state_to_idx = {s: i for i, s in enumerate(states)}
        obj._encoder = encoder
        return obj



# --- 3. VECTOR STATE SPACE (FIXED) ---
# --- 3. VECTOR STATE SPACE (FIXED) ---
class VectorStateSpace(AbstractDiscreteStateSpace):
    """
    Dynamic Vector Space with JAX broadcasting support.
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int):
        self.dim = dim
        self._matrix = jnp.zeros((0, dim), dtype=jnp.float32)

        unique = sorted(list(set(vectors)), key=lambda x: str(x))
        if unique:
            self._matrix = jnp.array([v.values for v in unique], dtype=np.float32)

        super().__init__(vectors, encoder=VectorEncoding(dim))

    def _raw_init_hook(self, raw_data, **kwargs):
        self.dim = kwargs.get('dim', raw_data.shape[1])
        self._matrix = jnp.array(raw_data, dtype=np.float32)
        if self._encoder is None:
            self._encoder = VectorEncoding(self.dim)

    def _on_state_added(self, state: VectorState):
        if len(self._idx_to_state) > self._matrix.shape[0]:
            new_vec = jnp.array(state.values, dtype=jnp.float32).reshape(1, self.dim)
            self._matrix = jnp.concatenate([self._matrix, new_vec], axis=0)

    # --- CRITICAL FIX: CONTAINS OVERRIDE ---
    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """
        Smart Contains for Vectors.
        Handles JAX Tracers safely without triggering Dictionary Lookups.
        """
        # 1. Check for JAX/Numpy Array (Data Check)
        # We check for hasattr('shape') to cover JAX Tracers, Numpy, and JAX Arrays safely
        if hasattr(state, 'shape') and hasattr(state, 'dtype'):

            # A. Vector Check (Float Type + Correct Dimension)
            is_float = jnp.issubdtype(state.dtype, jnp.floating) or jnp.issubdtype(state.dtype, jnp.complexfloating)

            # Handle Single Vector (D,) or Batch (..., D)
            if is_float and state.shape[-1] == self.dim:
                # Prepare shapes for broadcasting:
                # State: (..., 1, D) vs Matrix: (1, N, D)

                # If Single Vector (D,), reshape to (1, 1, D)
                if state.ndim == 1:
                    state_exp = state[None, None, :]
                else:
                    state_exp = state[..., None, :]

                matrix_exp = self._matrix[None, :, :]

                # Difference -> (..., N, D)
                diff = state_exp - matrix_exp
                matches = jnp.all(jnp.isclose(diff, 0, atol=1e-5), axis=-1)

                # Check if ANY state in matrix matches
                return jnp.any(matches, axis=-1)

            # B. Index Check (Integer Type)
            # If it's an array but NOT a vector match, we treat it as an Index ID check
            # (Valid if integer and within bounds)
            if jnp.issubdtype(state.dtype, jnp.integer):
                return (state >= 0) & (state < self.num_states)

            # C. Invalid Array Shape/Type for this Space
            # If we are here, we have an Array/Tracer that is neither a valid Vector
            # nor a valid Index. We return False (Array form) to satisfy JAX.
            return jnp.array(False)

        # 2. Fallback to Parent (Python Objects only)
        # Only reached if input is a pure Python object (State, List, etc.)
        return super().contains(state)

    def map(self, func: Callable[[Any], Any]) -> Union[jnp.ndarray, List[Any]]:
        if self._matrix is not None and self._matrix.shape[0] > 0:
            try:
                return jax.vmap(func)(self._matrix)
            except Exception:
                pass
        return super().map(func)

    def create_subset(self, states: List[VectorState]) -> 'VectorStateSpace':
        return VectorStateSpace(states, dim=self.dim)

    def get_matrix(self) -> jnp.ndarray:
        return self._matrix

    # --- Pytree ---
    def _tree_flatten(self):
        return (self._matrix,), (self._idx_to_state, self.dim, self._encoder)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        states, dim, encoder = aux
        matrix = children[0]
        obj = cls.__new__(cls)
        obj.dim = dim
        obj._idx_to_state = states
        obj._state_to_idx = {s: i for i, s in enumerate(states)}
        obj._encoder = encoder
        obj._matrix = matrix
        return obj

    def filter_by_index(self, axis_idx: int, value: float, atol: float = 1e-6) -> 'VectorStateSpace':
        """O(N) Linear Scan for non-indexed spaces."""
        matches = []
        for state in self._idx_to_state:
            if axis_idx < len(state.values):
                if abs(state.values[axis_idx] - value) <= atol:
                    matches.append(state)
        return self.create_subset(matches)

# --- 4. INDEXED VECTOR SPACE (FIXED) ---
class IndexedVectorStateSpace(VectorStateSpace):
    """
    An optimized VectorSpace that pre-builds lookup tables (Hash Maps).
    Trades Memory for Search Speed O(1).
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int, indexed_axes: tuple = (0,), precision: int = 6):
        self.indexed_axes = indexed_axes
        self.precision = precision

        # Containers
        self._value_to_index = {}
        self._axis_index_map = {axis: defaultdict(list) for axis in self.indexed_axes}

        super().__init__(vectors, dim=dim)

        # CRITICAL FIX: Explicitly populate the index after super().__init__
        # Because super().__init__ does bulk assignment, it skips _on_state_added calls.
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Populates the hash maps from the current state list."""
        for i, state in enumerate(self._idx_to_state):
            self._update_index_single(state, i)

    def _update_index_single(self, state: VectorState, idx: int):
        self._value_to_index[state.values] = idx
        vals = state.values
        for axis in self.indexed_axes:
            if axis < len(vals):
                key = round(vals[axis], self.precision)
                self._axis_index_map[axis][key].append(idx)

    def _on_state_added(self, state: VectorState):
        super()._on_state_added(state)
        # Handle dynamic addition
        idx = len(self._idx_to_state) - 1
        self._update_index_single(state, idx)

    # --- ADDED: SEARCH BY INDEX (O(1)) ---
    def search_by_index(self, axis_idx: int, value: float) -> 'VectorStateSpace':
        """O(1) Search using hash maps."""
        if axis_idx in self._axis_index_map:
            key = round(value, self.precision)
            target_indices = self._axis_index_map[axis_idx].get(key, [])

            if not target_indices:
                return self.create_subset([])

            subset_states = [self._idx_to_state[i] for i in target_indices]
            return self.create_subset(subset_states)

        # Fallback to linear search in parent
        return super().filter_by_index(axis_idx, value, atol=10 ** -self.precision)

    # --- ADDED: SELECT INDEX (Intersection) ---
    def select_index(self, axes: List[int], values: List[float]) -> 'VectorStateSpace':
        """Intersection Query across multiple axes."""
        if len(axes) != len(values): raise ValueError("Axes and Values must match length")

        candidate_indices = None
        manual_check = []

        # 1. Intersection of Indices (Fast)
        for axis, val in zip(axes, values):
            if axis in self._axis_index_map:
                key = round(val, self.precision)
                found = set(self._axis_index_map[axis].get(key, []))

                if candidate_indices is None:
                    candidate_indices = found
                else:
                    candidate_indices.intersection_update(found)

                if not candidate_indices:
                    return self.create_subset([])
            else:
                manual_check.append((axis, val))

        if candidate_indices is None:
            candidate_indices = range(self.num_states)

        # 2. Final Verification
        final_states = []
        for idx in candidate_indices:
            state = self._idx_to_state[idx]
            match = True
            for axis, val in manual_check:
                if not jnp.isclose(state.values[axis], val, atol=10 ** -self.precision):
                    match = False
                    break
            if match:
                final_states.append(state)

        return self.create_subset(final_states)