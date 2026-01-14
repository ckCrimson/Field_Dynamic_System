from dataclasses import dataclass
from collections import defaultdict
from typing import Set, List, Union, Sequence, Any, Optional

import jax.numpy as jnp

# Ensure these imports match your project structure
from .interfaces import IDiscreteStateSpace, StateEncoder, StateSpace, IStateOperation, State
from .encoding import BitMaskingEncoding, VectorEncoding
from .state import VectorState  # Assuming VectorState is defined in vector.py or similar


# --- Internal Helpers ---
@dataclass
class _BatchedVectorState:
    """Internal wrapper to allow applying operations on the entire JAX matrix at once."""
    values: jnp.ndarray


class AbstractDiscreteStateSpace(IDiscreteStateSpace):
    """
    A Finite Set of AbstractStates (e.g. {Rock, Paper, Scissor}).
    Uses Python Sets for fast O(1) membership checks.
    """

    def __init__(self,
                 states: Union[Set[Any], Sequence[Any]],
                 encoder: Optional[StateEncoder] = None):

        # 1. Deduplicate
        self.allowed_states = set(states)

        # 2. Sort for deterministic behavior (indices, encoding)
        # We try to sort; if elements aren't comparable, we use string representation
        try:
            self._sorted_states = sorted(list(self.allowed_states))
        except TypeError:
            self._sorted_states = sorted(list(self.allowed_states), key=lambda x: str(x))

        self._num_states = len(self._sorted_states)

        # 3. Encoder Strategy
        if encoder is None:
            self._encoder = BitMaskingEncoding(self._sorted_states)
        else:
            self._encoder = encoder

    # --- Properties Required by IDiscreteStateSpace ---

    @property
    def states(self) -> List[Any]:
        return self._sorted_states

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def is_empty(self) -> bool:
        return self._num_states == 0

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    @encoder.setter
    def encoder(self, new_encoder: StateEncoder):
        self._encoder = new_encoder

    # --- Core Methods ---

    def get_index_of(self, state_obj: Any) -> int:
        """
        Generic O(N) lookup.
        Subclasses should override this if they can do better.
        """
        try:
            return self._sorted_states.index(state_obj)
        except ValueError:
            return -1

    def create_subset(self, states: List[Any]) -> 'AbstractDiscreteStateSpace':
        """
        Factory: Creates a new instance of the same class containing only these states.
        """
        # Uses self.__class__ so VectorStateSpace returns VectorStateSpace
        return self.__class__(states)

    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """
        Checks membership.
        - Arrays: Checked as Integer IDs (valid if 0 <= id < num_states).
        - Objects/Lists: Checked against the internal Python Set.
        """
        # Case A: JAX Array (IDs)
        if isinstance(state, jnp.ndarray):
            return (state >= 0) & (state < self._num_states)

        # Case B: Python List (Batch)
        if isinstance(state, (list, tuple)):
            mask = [s in self.allowed_states for s in state]
            return jnp.array(mask, dtype=bool)

        # Case C: Single Object
        return state in self.allowed_states

    def get_matrix(self) -> jnp.ndarray:
        """Returns column vector of indices [0, 1, 2...]"""
        return jnp.arange(self._num_states, dtype=jnp.int32).reshape(-1, 1)

    # --- Set Operations ---

    def union(self, other: StateSpace) -> StateSpace:
        if isinstance(other, AbstractDiscreteStateSpace):
            new_set = self.allowed_states | other.allowed_states
            return self.create_subset(list(new_set))
        raise TypeError(f"Cannot union AbstractDiscreteStateSpace with {type(other)}")

    def intersection(self, other: StateSpace) -> StateSpace:
        if isinstance(other, AbstractDiscreteStateSpace):
            new_set = self.allowed_states & other.allowed_states
            return self.create_subset(list(new_set))
        raise TypeError(f"Cannot intersect AbstractDiscreteStateSpace with {type(other)}")

    def map(self, operation: IStateOperation) -> List[Any]:
        return [operation(s) for s in self._sorted_states]

    # --- JAX Flattening ---

    def _tree_flatten(self):
        children = ()
        aux_data = (self.allowed_states, self._encoder)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        allowed_states, encoder = aux_data
        obj = cls.__new__(cls)
        # Re-initialize basic properties manually to bypass __init__ cost
        obj.allowed_states = allowed_states
        # Re-sort to maintain consistency
        try:
            obj._sorted_states = sorted(list(allowed_states))
        except TypeError:
            obj._sorted_states = sorted(list(allowed_states), key=str)
        obj._num_states = len(obj._sorted_states)
        obj._encoder = encoder
        return obj


class VectorStateSpace(AbstractDiscreteStateSpace):
    """
    A Finite Set of specific VectorStates (Geometric Points).
    Optimized with JAX Matrices.
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int):
        # 1. Validation & Deduplication
        unique_vectors = tuple(set(vectors))

        # Validate dimensions
        for v in unique_vectors:
            if len(v.values) != dim:
                raise ValueError(f"Vector dim mismatch: expected {dim}, got {len(v.values)}")

        # 2. Initialize Parent
        super().__init__(list(unique_vectors))

        # 3. Specific Setup
        self.dim = dim
        self._encoder = VectorEncoding(dim)

        # Pre-compute matrix for fast JAX ops
        if self.allowed_states:
            # Note: allowed_states holds VectorState objects
            raw_list = [v.values for v in self._sorted_states]
            self._matrix = jnp.array(raw_list, dtype=jnp.float32)
        else:
            self._matrix = jnp.zeros((0, dim), dtype=jnp.float32)

    # --- Overrides for Vector Logic ---

    def create_subset(self, states: List[VectorState]) -> 'VectorStateSpace':
        """Required to pass 'dim' back to the constructor."""
        return VectorStateSpace(states, dim=self.dim)

    def get_index_of(self, state_obj: Any) -> int:
        """Optimized lookup: Identity first, then Value."""
        # 1. Identity Check
        idx = super().get_index_of(state_obj)
        if idx != -1:
            return idx

        # 2. Value Check (Fallback)
        target = state_obj.values if hasattr(state_obj, 'values') else state_obj

        # Check cache if available (from Indexed subclass)
        if hasattr(self, '_value_to_index'):
            return self._value_to_index.get(target, -1)

        # Linear scan
        for i, s in enumerate(self._sorted_states):
            if s.values == target:
                return i
        return -1

    def get_matrix(self) -> jnp.ndarray:
        """Returns the actual (N, D) coordinate matrix, not just indices."""
        return self._matrix

    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """Vectorized membership check."""
        if isinstance(state, jnp.ndarray):
            vecs = state
        else:
            # Attempt to encode python objects
            vecs = self.encoder.encode(state)

        if vecs.ndim == 1:
            vecs = vecs[None, :]

        if self._matrix.shape[0] == 0:
            return jnp.zeros(vecs.shape[0], dtype=bool)

        # Check distances
        tolerance = 1e-5
        # (M, 1, D) - (1, N, D) -> (M, N, D)
        diff = jnp.abs(vecs[:, None, :] - self._matrix[None, :, :])
        match = jnp.all(diff < tolerance, axis=-1)
        is_valid = jnp.any(match, axis=-1)

        if is_valid.shape == (1,):
            return is_valid[0]
        return is_valid

    def union(self, other: StateSpace) -> StateSpace:
        if isinstance(other, VectorStateSpace):
            if self.dim != other.dim:
                raise ValueError(f"Dim mismatch: {self.dim} vs {other.dim}")
            # Use parent logic, but wrap in proper constructor
            new_set = self.allowed_states | other.allowed_states
            return VectorStateSpace(list(new_set), self.dim)
        raise TypeError(f"Cannot union VectorStateSpace with {type(other)}")

    def intersection(self, other: StateSpace) -> StateSpace:
        if isinstance(other, VectorStateSpace):
            if self.dim != other.dim:
                raise ValueError(f"Dim mismatch: {self.dim} vs {other.dim}")
            new_set = self.allowed_states & other.allowed_states
            return VectorStateSpace(list(new_set), self.dim)
        raise TypeError(f"Cannot intersect VectorStateSpace with {type(other)}")

    def map(self, operation: IStateOperation) -> Any:
        # Optimized: Pass the whole matrix to the operation
        batched = _BatchedVectorState(self._matrix)
        return operation(batched)

    def filter_by_index(self, axis_idx: int, value: float, atol: float = 1e-6) -> 'VectorStateSpace':
        """Returns subset where vector[axis] ~= value."""
        if self._matrix.shape[0] == 0:
            return self.create_subset([])

        column = self._matrix[:, axis_idx]
        mask = jnp.isclose(column, value, atol=atol)

        # We need the objects corresponding to the rows
        filtered_states = [self._sorted_states[i] for i in range(len(mask)) if mask[i]]
        return self.create_subset(filtered_states)

    # --- JAX Pytree Registration ---
    def _tree_flatten(self):
        children = (self._matrix,)
        # Must store dim to reconstruct
        aux_data = (self.allowed_states, self.dim, self._encoder)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        allowed_states, dim, encoder = aux_data
        matrix = children[0]

        obj = cls.__new__(cls)
        # Restore AbstractDiscreteStateSpace properties
        obj.allowed_states = allowed_states
        # Ensure sorting logic matches init
        try:
            obj._sorted_states = sorted(list(allowed_states))
        except TypeError:
            obj._sorted_states = sorted(list(allowed_states), key=str)
        obj._num_states = len(obj._sorted_states)

        # Restore VectorStateSpace properties
        obj.dim = dim
        obj._encoder = encoder
        obj._matrix = matrix
        return obj


class IndexedVectorStateSpace(VectorStateSpace):
    """
    An optimized VectorSpace that pre-builds lookup tables (Hash Maps).
    Trades Memory for Search Speed O(1).
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int, indexed_axes: tuple = (0,), precision: int = 6):
        super().__init__(vectors, dim=dim)

        self.indexed_axes = indexed_axes
        self.precision = precision

        # Build Index (Value Tuple -> Int Index) for get_index_of
        self._value_to_index = {v.values: i for i, v in enumerate(self._sorted_states)}

        # Build Axis Index (Axis Value -> List of Int Indices)
        self._axis_index_map = self._build_axis_index()

    def _build_axis_index(self) -> dict:
        lookup = {axis: defaultdict(list) for axis in self.indexed_axes}
        for i, state in enumerate(self._sorted_states):
            vals = state.values
            for axis in self.indexed_axes:
                if axis < len(vals):
                    key = round(vals[axis], self.precision)
                    lookup[axis][key].append(i)
        return lookup

    def search_by_index(self, axis_idx: int, value: float) -> 'VectorStateSpace':
        """O(1) Search."""
        if axis_idx in self._axis_index_map:
            key = round(value, self.precision)
            target_indices = self._axis_index_map[axis_idx].get(key, [])

            if not target_indices:
                return self.create_subset([])

            subset_states = [self._sorted_states[i] for i in target_indices]
            return self.create_subset(subset_states)

        # Fallback
        return super().filter_by_index(axis_idx, value, atol=10 ** -self.precision)

    def select_index(self, axes: List[int], values: List[float]) -> 'VectorStateSpace':
        """Intersection Query across multiple axes."""
        if len(axes) != len(values):
            raise ValueError("Axes and Values must match length")

        candidate_indices = None
        manual_check = []

        # 1. Intersection of Indices
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
            state = self._sorted_states[idx]
            match = True
            for axis, val in manual_check:
                if not jnp.isclose(state.values[axis], val, atol=10 ** -self.precision):
                    match = False
                    break
            if match:
                final_states.append(state)

        return self.create_subset(final_states)