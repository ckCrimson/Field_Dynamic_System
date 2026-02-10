from dataclasses import dataclass
from collections import defaultdict
from typing import Set, List, Union, Sequence, Any, Optional, Callable, Type

import jax.numpy as jnp
import jax
import numpy as np

# Ensure these imports match your project structure
from .interfaces import IDiscreteStateSpace, StateEncoder, StateSpace, State
from .encoding import BitMaskingEncoding, VectorEncoding, LazyEncoder
from .state import VectorState


# --- 1. THE LAZY PROXY ---
# --- 1. LAZY PROXY ---
class LazyStateProxy:
    """
    Universal Proxy: Wraps raw data + optional IDs.
    """
    def __init__(self, raw_collection: Sequence[Any], wrapper_func: Callable[[Any], Any], ids: Optional[Sequence[int]] = None):
        self.raw_data = raw_collection
        self._wrapper = wrapper_func
        self._ids = ids

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        val = self.raw_data[idx]
        if self._ids is not None:
            return self._wrapper(val, id=self._ids[idx])
        return self._wrapper(val)

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


    def _raw_init_hook(self, raw_data, **kwargs):
        pass

    # --- Standard Properties ---

    @property
    def num_states(self) -> int:
        return len(self._idx_to_state)

    @property
    def state_class(self) -> Type:
        """
        Returns the class used to wrap raw data into State Objects.
        Used by Transformations to reconstruct standard spaces.
        """
        # 1. If Lazy Proxy exists, it holds the wrapper reference
        if hasattr(self._idx_to_state, '_wrapper'):
            return self._idx_to_state._wrapper

        # 2. If objects exist, inspect the first one
        if self._idx_to_state and len(self._idx_to_state) > 0:
            return type(self._idx_to_state[0])

        # 3. Default Fallback (Import locally to avoid circular deps if needed)
        from src.field_dynamic_system.core.state import AbstractState
        return AbstractState

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

    def get_state_by_id(self, idx: Union[int, jnp.ndarray, np.ndarray]) -> Any:
        """
        Retrieves the State object corresponding to a numeric ID.

        Args:
            idx: The integer index (can be Python int, NumPy int, or JAX scalar).

        Returns:
            The actual State object.
        """
        # 1. Normalize Input (Handle JAX/NumPy scalars via .item())
        # This fixes errors where 'idx' is a JAX array from argmax
        if hasattr(idx, 'item'):
            i = int(idx.item())
        else:
            i = int(idx)

        # 2. Bounds Check
        if i < 0 or i >= len(self._idx_to_state):
            raise IndexError(f"State ID {i} out of bounds. Space size: {len(self._idx_to_state)}")

        # 3. Lookup
        return self._idx_to_state[i]

    @classmethod
    def from_raw_data(cls, raw_data: Sequence[Any], wrapper: Callable[[Any], Any], ids: Optional[Sequence[int]] = None,
                      **kwargs):
        obj = cls.__new__(cls)
        obj._idx_to_state = LazyStateProxy(raw_data, wrapper, ids)
        obj._state_to_idx = {}
        obj._encoder = kwargs.get('encoder')
        obj._raw_init_hook(raw_data, **kwargs)
        return obj

    # --- RAW LOOKUPS (Optimization: Lazy Raw Map) ---

    def _ensure_raw_map(self):
        """
        Lazily builds a {RawValue -> Index} map.
        This enables O(1) lookups for raw strings/numbers without object overhead.
        """
        # Check if map already exists
        if hasattr(self, '_raw_map') and self._raw_map is not None:
            return

        # Build Map: Iterate raw buffer directly
        # This is fast because we deal with primitives (str/int), not State objects.
        try:
            # We assume raw_states returns a Numpy/JAX array
            raw = self.raw_states
            # Convert to standard Python types for Dictionary keys (Numpy scalars can be slow as keys)
            if hasattr(raw, 'tolist'):
                raw = raw.tolist()

            self._raw_map = {val: i for i, val in enumerate(raw)}
        except Exception:
            # Fallback for complex types (like Vectors) that aren't hashable directly
            # For Vectors, we'd need tuple conversion, handled by specific subclasses
            self._raw_map = {}

    def contains_raw(self, raw_val: Any) -> bool:
        """
        O(1) Check if raw value exists.
        """
        # 1. Try Lazy Map (Fastest)
        self._ensure_raw_map()
        if self._raw_map:
            return raw_val in self._raw_map

        # 2. Fallback to Array Scan (Slow, but safe)
        # (This was the logic giving you 0.0x speedup)
        return raw_val in self.raw_states

    def get_raw_index(self, raw_val: Any) -> int:
        """
        O(1) Get Index from Raw Value.
        """
        self._ensure_raw_map()
        return self._raw_map.get(raw_val, -1)

    def _build_raw_map(self):
        """
        Builds {RawValue -> Int} map.
        Much lighter than {Object -> Int}.
        """
        # We access the raw buffer directly
        raw = self.raw_states
        # Create map from raw primitives (int/str) to index
        # This is fast because we don't inflate Objects.
        self._raw_map = {val: i for i, val in enumerate(raw)}

    def _decode_index(self, raw_val):
        """Override this in subclasses for math-based lookup."""
        return None



    # --- RAW SET OPERATIONS (The Speed Boost) ---

    @property
    def raw_states(self) -> np.ndarray:
        if isinstance(self._idx_to_state, LazyStateProxy):
            return np.array(self._idx_to_state.raw_data)
        return np.array(self.get_matrix())  # Fallback

    # --- RAW SET OPERATIONS (1D Generic) ---
    def raw_union(self, other_space):
        # np.union1d works for Strings and Scalars
        res = np.union1d(self.raw_states, other_space.raw_states)
        return self.from_raw_data(res, self._get_wrapper_cls())

    def raw_intersection(self, other_space):
        res = np.intersect1d(self.raw_states, other_space.raw_states)
        return self.from_raw_data(res, self._get_wrapper_cls())

    def _get_wrapper_cls(self):
        if isinstance(self._idx_to_state, LazyStateProxy):
            return self._idx_to_state._wrapper
        return type(self._idx_to_state[0]) if self._idx_to_state else None


# ----3. Lazy Discrete State Space -----------#
# --- HELPER: Fast Row-wise Set Operations ---
def _view_as_void(arr):
    """
    Tricks NumPy into treating each row (N, D) as a single element.
    Essential for fast set operations (union/intersection) on matrices.
    """
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))


def _unique_rows(arr):
    """Returns unique rows from a matrix."""
    if arr.size == 0: return arr
    _, idx = np.unique(_view_as_void(arr), return_index=True)
    return arr[np.sort(idx)]


def _intersect_matrices(A, B):
    """Finds rows present in both A and B."""
    if A.size == 0 or B.size == 0:
        return np.empty((0, A.shape[1]), dtype=A.dtype)

    voidA = _view_as_void(A)
    voidB = _view_as_void(B)

    # 1D Intersection on void views
    # intersect1d returns sorted unique elements
    common_void = np.intersect1d(voidA, voidB)

    # Convert back to original type
    return common_void.view(A.dtype).reshape(-1, A.shape[1])


class LazyDiscreteStateSpace(IDiscreteStateSpace):
    """
    Data-Oriented State Space.
    Supports Set Operations directly on the memory buffer.
    """

    def __init__(self, raw_data: Union[np.ndarray, jnp.ndarray], wrapper_class: Callable):
        # Enforce 2D shape (N, D) even for 1D inputs
        self._raw_data = np.array(raw_data)
        if self._raw_data.ndim == 1:
            self._raw_data = self._raw_data.reshape(-1, 1)

        self._wrapper = wrapper_class
        self._size = self._raw_data.shape[0]
        self._encoder = LazyEncoder()

    # --- 1. SET OPERATIONS (Raw Data Mode) ---

    def union(self, other: 'StateSpace') -> 'LazyDiscreteStateSpace':
        """
        Returns a NEW LazySpace combining rows from both spaces.
        Operation: Stack -> Unique.
        """
        if not isinstance(other, LazyDiscreteStateSpace):
            raise TypeError("Lazy union requires another LazyDiscreteStateSpace.")

        # 1. Stack Data
        combined = np.vstack((self._raw_data, other._raw_data))

        # 2. Filter Duplicates (Fast Raw Logic)
        unique_combined = _unique_rows(combined)

        # 3. Return New Space
        return LazyDiscreteStateSpace(unique_combined, self._wrapper)

    def intersection(self, other: 'StateSpace') -> 'LazyDiscreteStateSpace':
        """
        Returns a NEW LazySpace with rows common to both.
        Operation: View Void -> Intersect1d.
        """
        if not isinstance(other, LazyDiscreteStateSpace):
            raise TypeError("Lazy intersection requires another LazyDiscreteStateSpace.")

        # 1. Fast Matrix Intersection
        common_data = _intersect_matrices(self._raw_data, other._raw_data)

        # 2. Return New Space
        return LazyDiscreteStateSpace(common_data, self._wrapper)

    def create_subset(self, indices_or_states: Sequence[Any]) -> 'LazyDiscreteStateSpace':
        """
        Creates a Subset Space.
        Optimized to accept a list of Indices (Integers) to slice the matrix.
        """
        # If input is indices (Fast Slicing)
        if isinstance(indices_or_states, (np.ndarray, slice, list)) and \
                (isinstance(indices_or_states, slice) or np.issubdtype(np.array(indices_or_states).dtype, np.integer)):
            subset_data = self._raw_data[indices_or_states]
            return LazyDiscreteStateSpace(subset_data, self._wrapper)

        raise NotImplementedError("For Lazy Spaces, create_subset requires Indices (int array).")

    # --- 2. TRANSFORMATION (Map) ---

    def map_raw(self, func: Callable[[np.ndarray], np.ndarray],
                new_wrapper: Callable = None) -> 'LazyDiscreteStateSpace':
        """
        Vectorized Map: Transforms the underlying matrix directly.

        Args:
            func: A function f(Matrix_A) -> Matrix_B
            new_wrapper: Optional new wrapper if dimensionality changes.
        """
        # Apply vector function to entire matrix
        new_data = func(self._raw_data)

        # Use new wrapper if provided, else inherit
        wrap = new_wrapper if new_wrapper else self._wrapper

        return LazyDiscreteStateSpace(new_data, wrap)

    def map(self, func: Callable[[Any], Any]):
        """Standard Object Map (Slow - exists for compatibility)."""
        results = []
        for i in range(self._size):
            s = self.get_state_by_id(i)
            results.append(func(s))
        return results

    # --- 3. STANDARD PROTOCOL METHODS ---

    @property
    def num_states(self) -> int:
        return self._size

    def size(self) -> int:
        return self._size

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    def get_matrix(self) -> jnp.ndarray:
        return jnp.array(self._raw_data)

    def get_state_by_id(self, idx: Union[int, jnp.ndarray]) -> Any:
        if hasattr(idx, 'item'):
            i = int(idx.item())
        else:
            i = int(idx)
        if i < 0 or i >= self._size: raise IndexError(f"Index {i} out of bounds.")

        row = self._raw_data[i]
        if row.size == 1: return self._wrapper(row.item())  # Handle scalar wrapper
        return self._wrapper(*row)

    def get_index_of(self, state_obj: Any) -> int:
        raise NotImplementedError("Use Raw Data lookups.")

    def register_states(self, states_batch: Sequence[Any]) -> jnp.ndarray:
        if isinstance(states_batch, (np.ndarray, jnp.ndarray, range)):
            return jnp.array(states_batch, dtype=jnp.int32)
        if len(states_batch) > 0 and isinstance(states_batch[0], (int, np.integer)):
            return jnp.array(states_batch, dtype=jnp.int32)
        raise NotImplementedError("Use Indices for Lazy Space registration.")

    def contains(self, state):
        if isinstance(state, (int, np.integer)): return 0 <= state < self._size
        return False

    def ids_view(self):
        return range(self._size)

    @property
    def states(self):
        return [self.get_state_by_id(i) for i in range(self._size)]

    # Pytree support
    def _tree_flatten(self):
        return (self._raw_data,), (self._wrapper,)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(children[0], aux[0])

# --- 4. VECTOR STATE SPACE  ---
def _view_as_void(arr):
    """Tricks NumPy into treating rows as single elements."""
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

def _unique_rows(arr):
    if arr.size == 0: return arr
    _, idx = np.unique(_view_as_void(arr), return_index=True)
    return arr[np.sort(idx)]

def _intersect_rows(A, B):
    if A.size == 0 or B.size == 0: return np.empty((0, A.shape[1]), dtype=A.dtype)
    voidA, voidB = _view_as_void(A), _view_as_void(B)
    common = np.intersect1d(voidA, voidB)
    return common.view(A.dtype).reshape(-1, A.shape[1])

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
        # [FIX 2] Handle 1D vs 2D raw data
        # If raw_data came from union1d (Bad), it would be 1D.
        # But we fixed raw_union below, so raw_data should be 2D here.

        if hasattr(raw_data, 'ndim') and raw_data.ndim == 1:
            # Emergency reshape if single vector passed as list
            pass

        self.dim = kwargs.get('dim', raw_data.shape[1])
        self._matrix = jnp.array(raw_data, dtype=np.float32)
        if self._encoder is None:
            self._encoder = VectorEncoding(self.dim)

    # --- [FIX 2] OVERRIDE RAW OPS FOR 2D DATA ---
    def raw_union(self, other):
        """Row-wise Union."""
        A = np.array(self._matrix)
        B = np.array(other._matrix)
        combined = np.vstack((A, B))
        unique = _unique_rows(combined)

        return self.from_raw_data(unique, self._get_wrapper_cls(), dim=self.dim)

    def raw_intersection(self, other):
        """Row-wise Intersection."""
        A = np.array(self._matrix)
        B = np.array(other._matrix)
        common = _intersect_rows(A, B)

        return self.from_raw_data(common, self._get_wrapper_cls(), dim=self.dim)

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

    @property
    def state_class(self) -> Type:
        from src.field_dynamic_system.core.state import VectorState
        return VectorState

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


# --- 5. INDEXED VECTOR SPACE ---
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

    def _raw_init_hook(self, raw_data, **kwargs):
        # 1. Call Parent Hook (Sets up .dim, ._matrix)
        super()._raw_init_hook(raw_data, **kwargs)

        # 2. Setup Indexed Configuration
        self.indexed_axes = kwargs.get('indexed_axes', (0,))
        self.precision = kwargs.get('precision', 6)

        # 3. Initialize Containers
        self._value_to_index = {}
        self._axis_index_map = {axis: defaultdict(list) for axis in self.indexed_axes}

        # 4. Build Indices
        # Warning: This iterates all states.
        # For HUGE raw data, you might want to delay this (lazy indexing),
        # but for IndexedSpace the promise is O(1) search, so we usually build eagerly.

        # Optimization: We can index the _matrix directly without inflating objects!
        self._build_indices_from_matrix(self._matrix)

    def _build_indices_from_matrix(self, matrix):
        """
        Fast Indexing using Raw Matrix (Avoids Object Inflation).
        """
        for i in range(matrix.shape[0]):
            row = matrix[i]
            # Convert JAX/Numpy row to tuple for Dict Key
            # We use distinct values for precise lookup
            val_key = tuple(np.array(row).tolist())
            self._value_to_index[val_key] = i

            # Axis Indexing
            for axis in self.indexed_axes:
                if axis < len(row):
                    key = round(float(row[axis]), self.precision)
                    self._axis_index_map[axis][key].append(i)

