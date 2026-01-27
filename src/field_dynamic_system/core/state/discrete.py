from dataclasses import dataclass
from collections import defaultdict
from typing import Set, List, Union, Sequence, Any, Optional

import jax.numpy as jnp

# Ensure these imports match your project structure
from .interfaces import IDiscreteStateSpace, StateEncoder, StateSpace, IStateOperation, State
from .encoding import BitMaskingEncoding, VectorEncoding
from .state import VectorState


@dataclass
class _BatchedVectorState:
    values: jnp.ndarray


class AbstractDiscreteStateSpace(IDiscreteStateSpace):
    """
    Dynamic State Space.
    Maintains a bijection between State Objects and Integer Indices.
    """

    def __init__(self,
                 states: Union[Set[Any], Sequence[Any]],
                 encoder: Optional[StateEncoder] = None):

        # 1. Internal Storage
        # Convert to list and sort for deterministic index assignment
        # Using str(x) is a safe generic sort key
        initial_list = sorted(list(set(states)), key=lambda x: str(x))

        self._idx_to_state = []
        self._state_to_idx = {}

        # 2. Encoder
        self._encoder = encoder if encoder else BitMaskingEncoding(initial_list)

        # 3. Populate (Uses register_states to ensure _on_state_added is called)
        self.register_states(initial_list)

    @property
    def num_states(self) -> int:
        return len(self._idx_to_state)

    @property
    def states(self) -> List[Any]:
        return self._idx_to_state

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    # --- Dynamic Registration ---

    def register_states(self, states_batch: Sequence[Any]) -> jnp.ndarray:
        indices = []

        for s in states_batch:
            # O(1) Lookup - This prevents the N^2 bottleneck
            if s not in self._state_to_idx:
                new_idx = len(self._idx_to_state)
                self._state_to_idx[s] = new_idx
                self._idx_to_state.append(s)

                # Hook for subclasses (Matrix update / Index update)
                self._on_state_added(s)

            indices.append(self._state_to_idx[s])

        return jnp.array(indices, dtype=jnp.int32)

    def add_state(self, state: Any) -> int:
        return int(self.register_states([state])[0])

    def _on_state_added(self, state):
        """Override in subclasses to handle new state registration."""
        pass

    # --- Lookup ---

    def get_index_of(self, state_obj: Any) -> int:
        return self._state_to_idx.get(state_obj, -1)

    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        if isinstance(state, jnp.ndarray):
            return (state >= 0) & (state < self.num_states)
        if isinstance(state, (list, tuple)):
            return jnp.array([s in self._state_to_idx for s in state], dtype=bool)
        return state in self._state_to_idx

    # --- Set Ops ---
    def create_subset(self, states: List[Any]) -> 'AbstractDiscreteStateSpace':
        return self.__class__(states)

    def union(self, other: StateSpace) -> StateSpace:
        if isinstance(other, AbstractDiscreteStateSpace):
            combined = list(set(self.states) | set(other.states))
            return self.create_subset(combined)
        raise TypeError(f"Cannot union with {type(other)}")

    # --- Flattening ---
    def _tree_flatten(self):
        return (), (self._idx_to_state, self._encoder)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        states, encoder = aux
        obj = cls.__new__(cls)
        obj._idx_to_state = []
        obj._state_to_idx = {}
        obj._encoder = encoder
        # Re-register to rebuild lookups
        obj.register_states(states)
        return obj


class VectorStateSpace(AbstractDiscreteStateSpace):
    """
    Dynamic Vector Space.
    Updates the JAX matrix buffer when new vectors are registered.
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int):
        self.dim = dim
        self._matrix = jnp.zeros((0, dim), dtype=jnp.float32)
        # super init calls register_states -> _on_state_added
        super().__init__(vectors, encoder=VectorEncoding(dim))

    # def _on_state_added(self, state: VectorState):
    #     """Called automatically by register_states when a new vector comes in."""
    #     new_vec = jnp.array(state.values, dtype=jnp.float32).reshape(1, self.dim)
    #     # Concatenate to keep matrix in sync with _idx_to_state
    #     self._matrix = jnp.concatenate([self._matrix, new_vec], axis=0)

    def _on_state_added(self, state: VectorState):
        """
        Only append if the matrix size is actually smaller than the state count.
        If we pre-allocated, this becomes a no-op or a simple check.
        """
        if len(self._idx_to_state) > self._matrix.shape[0]:
            new_vec = jnp.array(state.values, dtype=jnp.float32).reshape(1, self.dim)
            self._matrix = jnp.concatenate([self._matrix, new_vec], axis=0)

    def create_subset(self, states: List[VectorState]) -> 'VectorStateSpace':
        return VectorStateSpace(states, dim=self.dim)

    def get_matrix(self) -> jnp.ndarray:
        return self._matrix

    # --- Missing Method Implementation (Linear Fallback) ---
    def filter_by_index(self, axis_idx: int, value: float, atol: float = 1e-6) -> 'VectorStateSpace':
        """
        Linear search fallback O(N).
        Used if IndexedVectorStateSpace lookup fails or for unindexed axes.
        """
        matches = []
        for state in self._idx_to_state:
            # Check axis bounds
            if axis_idx < len(state.values):
                if abs(state.values[axis_idx] - value) <= atol:
                    matches.append(state)
        return self.create_subset(matches)

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
        # Rebuild dictionary lookup
        obj._state_to_idx = {s: i for i, s in enumerate(states)}
        obj._encoder = encoder
        obj._matrix = matrix
        return obj


class IndexedVectorStateSpace(VectorStateSpace):
    """
    An optimized VectorSpace that pre-builds lookup tables (Hash Maps).
    Trades Memory for Search Speed O(1).
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int, indexed_axes: tuple = (0,), precision: int = 6):
        self.indexed_axes = indexed_axes
        self.precision = precision

        # Initialize containers BEFORE calling super().__init__
        # because super() calls register_states -> _on_state_added immediately
        self._value_to_index = {}
        self._axis_index_map = {axis: defaultdict(list) for axis in self.indexed_axes}

        super().__init__(vectors, dim=dim)

    def _on_state_added(self, state: VectorState):
        """
        Override to update indexes dynamically.
        CRITICAL: Must call super() to update the JAX Matrix.
        """
        # 1. Update Matrix (Parent logic)
        super()._on_state_added(state)

        # 2. Get the new index (it was just added to the end of the list)
        idx = len(self._idx_to_state) - 1

        # 3. Update Value Lookup (Tuple -> Index)
        self._value_to_index[state.values] = idx

        # 4. Update Axis Indexes
        vals = state.values
        for axis in self.indexed_axes:
            if axis < len(vals):
                key = round(vals[axis], self.precision)
                self._axis_index_map[axis][key].append(idx)

    def search_by_index(self, axis_idx: int, value: float) -> 'VectorStateSpace':
        """O(1) Search using hash maps."""
        if axis_idx in self._axis_index_map:
            key = round(value, self.precision)
            target_indices = self._axis_index_map[axis_idx].get(key, [])

            if not target_indices:
                return self.create_subset([])

            # FIX: Use self._idx_to_state, not self._sorted_states
            subset_states = [self._idx_to_state[i] for i in target_indices]
            return self.create_subset(subset_states)

        # Fallback to linear search in parent
        return super().filter_by_index(axis_idx, value, atol=10 ** -self.precision)

    def select_index(self, axes: List[int], values: List[float]) -> 'VectorStateSpace':
        """Intersection Query across multiple axes."""
        if len(axes) != len(values):
            raise ValueError("Axes and Values must match length")

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
            # If no axes were indexed, scan everything
            candidate_indices = range(self.num_states)

        # 2. Final Verification (for unindexed axes or precision collision)
        final_states = []
        for idx in candidate_indices:
            # FIX: Use self._idx_to_state
            state = self._idx_to_state[idx]
            match = True
            for axis, val in manual_check:
                if not jnp.isclose(state.values[axis], val, atol=10 ** -self.precision):
                    match = False
                    break
            if match:
                final_states.append(state)

        return self.create_subset(final_states)