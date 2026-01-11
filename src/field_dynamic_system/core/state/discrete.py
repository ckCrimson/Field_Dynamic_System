from dataclasses import dataclass, field

import jax.numpy as jnp
from .interfaces import IDiscreteStateSpace, StateEncoder, StateSpace, State, IStateOperation
from .encoding import BitMaskingEncoding, VectorEncoding
from .state import VectorState, AbstractState
from collections import defaultdict

from typing import Set, List, Union, Sequence, Any, Optional


class AbstractDiscreteStateSpace(IDiscreteStateSpace):
    """
    A Finite Set of AbstractStates (e.g. {Rock, Paper, Scissor}).
    Uses Set theory for fast lookups.
    """

    def __init__(self,
                 states: Union[Set[AbstractState], Sequence[AbstractState]],
                 encoder: Optional[StateEncoder] = None):

        self.allowed_states = set(states)

        # Optimization: Sort once for deterministic iteration/encoding
        self._sorted_states = sorted(list(self.allowed_states), key=lambda x: str(x))
        self._num_states = len(self._sorted_states)

        # Dependency Injection: Use provided encoder or default to BitMasking
        if encoder is None:
            self._encoder = BitMaskingEncoding(self._sorted_states)
        else:
            self._encoder = encoder

    @property
    def num_states(self) -> int:
        """Returns the number of unique symbolic states."""
        return len(self.allowed_states)

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    @encoder.setter
    def encoder(self, new_encoder: StateEncoder):
        """
        Allows hot-swapping the encoder strategy.
        Note: The user is responsible for ensuring the new encoder
        is compatible with the current set of states.
        """
        self._encoder = new_encoder

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    def get_matrix(self) -> jnp.ndarray:
        # Returns [0, 1, 2, ...]
        return jnp.arange(self._num_states, dtype=jnp.int32).reshape(-1, 1)

    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """
        Checks membership.
        - Objects/Lists: Checked against the Python Set (O(1)).
        - Arrays: Checked as Integer IDs (valid if 0 <= id < num_states).
        """
        # Case A: JAX Array (IDs)
        if isinstance(state, jnp.ndarray):
            # If we get raw integers, we assume they are IDs relative to this encoder.
            # Valid IDs are 0 to N-1.
            return (state >= 0) & (state < self._num_states)

        # Case B: Python List (Batch)
        if isinstance(state, list) or isinstance(state, tuple):
            # Python Set lookup is faster than JAX for abstract objects
            # Returns boolean array
            mask = [s in self.allowed_states for s in state]
            return jnp.array(mask, dtype=bool)

        # Case C: Single Object
        return state in self.allowed_states

    def union(self, other: StateSpace) -> StateSpace:
        """
        Combines two sets of states.
        Example: {Red, Green} U {Blue} = {Red, Green, Blue}
        """
        if isinstance(other, AbstractDiscreteStateSpace):
            # Set Union: | operator
            new_set = self.allowed_states | other.allowed_states
            return AbstractDiscreteStateSpace(new_set)

        raise TypeError(f"Cannot union AbstractDiscreteStateSpace with {type(other)}")

    def intersection(self, other: StateSpace) -> StateSpace:
        """
        Keeps only states present in BOTH spaces.
        Example: {Red, Green} ∩ {Green, Blue} = {Green}
        """
        if isinstance(other, AbstractDiscreteStateSpace):
            # Set Intersection: & operator
            new_set = self.allowed_states & other.allowed_states
            return AbstractDiscreteStateSpace(new_set)

        raise TypeError(f"Cannot intersect AbstractDiscreteStateSpace with {type(other)}")

    # Inside AbstractDiscreteStateSpace class...

    def map(self, operation: IStateOperation) -> List[Any]:
        """
        Applies an operation to all abstract states.
        Uses the pre-sorted cached list for O(N) speed.
        """
        # FAST PATH: No sorting, just iteration.
        return [operation(s) for s in self._sorted_states]

    def _tree_flatten(self):
        # SPECIALIZED: No arrays here, so children is empty.
        children = ()
        # Everything is metadata
        aux_data = (self.allowed_states, self._encoder, self._num_states)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        allowed_states, encoder, num_states = aux_data

        # BYPASS INIT
        obj = cls.__new__(cls)
        obj.allowed_states = allowed_states
        # We can re-sort cheaply or store the sorted list in aux_data if preferred
        obj._sorted_states = sorted(list(allowed_states), key=lambda x: str(x))
        obj._encoder = encoder
        obj._num_states = num_states
        return obj


@dataclass
class _BatchedVectorState:
    """Internal wrapper for JAX optimizations."""
    values: jnp.ndarray


class VectorStateSpace(IDiscreteStateSpace):
    """
    A Finite Set of specific VectorStates.
    Standard Python Class.
    """

    def __init__(self, vectors: Sequence[VectorState], dim: int):
        # We convert to a set immediately to remove duplicates
        self.allowed_vectors = tuple(set(vectors))

        # FIX: Use 'self.dim' (public) so children and external calls can see it
        self.dim = dim

        self._encoder = VectorEncoding(dim)

        # Pre-compute matrix
        if self.allowed_vectors:
            raw_list = [v.values for v in self.allowed_vectors]
            self._matrix = jnp.array(raw_list, dtype=jnp.float32)
        else:
            self._matrix = jnp.zeros((0, dim), dtype=jnp.float32)

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    def get_matrix(self) -> jnp.ndarray:
        return self._matrix

    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        if not isinstance(state, jnp.ndarray):
            vecs = self.encoder.encode(state)
        else:
            vecs = state

        if vecs.ndim == 1:
            vecs = vecs[None, :]

        if self._matrix.shape[0] == 0:
            res = jnp.zeros(vecs.shape[0], dtype=bool)
            return res[0] if res.shape == (1,) else res

        tolerance = 1e-5
        diff = jnp.abs(vecs[:, None, :] - self._matrix[None, :, :])
        match_matrix = jnp.all(diff < tolerance, axis=-1)
        is_valid = jnp.any(match_matrix, axis=-1)

        if is_valid.shape == (1,):
            return is_valid[0]
        return is_valid

    def union(self, other: StateSpace) -> StateSpace:
        if isinstance(other, VectorStateSpace):
            # FIX: Use self.dim
            if self.dim != other.dim:
                raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")

            new_set = set(self.allowed_vectors).union(set(other.allowed_vectors))
            return VectorStateSpace(list(new_set), self.dim)

        raise TypeError(f"Cannot union VectorStateSpace with {type(other)}")

    def intersection(self, other: StateSpace) -> StateSpace:
        if isinstance(other, VectorStateSpace):
            # FIX: Use self.dim
            if self.dim != other.dim:
                raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")

            new_set = set(self.allowed_vectors).intersection(set(other.allowed_vectors))
            return VectorStateSpace(list(new_set), self.dim)

        raise TypeError(f"Cannot intersect VectorStateSpace with {type(other)}")

    def map(self, operation: IStateOperation) -> Any:
        batched_state = _BatchedVectorState(self._matrix)
        return operation(batched_state)

    # --- JAX Pytree Registration ---
    def _tree_flatten(self):
        children = (self._matrix,)
        # FIX: Store self.dim in aux_data
        aux_data = (self.allowed_vectors, self.dim, self._encoder)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        allowed_vectors, dim, encoder = aux_data
        matrix = children[0]

        obj = cls.__new__(cls)
        obj.allowed_vectors = allowed_vectors
        # FIX: Restore to self.dim
        obj.dim = dim
        obj._encoder = encoder
        obj._matrix = matrix
        return obj

    @property
    def num_states(self) -> int:
        return len(self.allowed_vectors)

    def filter_by_index(self, axis_idx: int, value: float, atol: float = 1e-6) -> 'VectorStateSpace':
        """
        Returns a new VectorStateSpace containing only vectors where
        vector[axis_idx] is approximately equal to 'value'.
        """
        # 1. Extract the relevant column from the JAX matrix
        column = self._matrix[:, axis_idx]

        # 2. Create a Boolean Mask (Fast JAX comparison)
        mask = jnp.isclose(column, value, atol=atol)

        # 3. Apply mask to get the subset of raw vectors
        filtered_data = self._matrix[mask]

        # 4. Handle Empty Result
        if filtered_data.shape[0] == 0:
            return VectorStateSpace([], dim=self.dim)

        # 5. Reconstruct the Subspace
        # FIX: Use .tolist() to convert JAX Arrays -> Python Floats
        new_vectors = [VectorState(tuple(row.tolist())) for row in filtered_data]

        return VectorStateSpace(new_vectors, dim=self.dim)



class IndexedVectorStateSpace(VectorStateSpace):
    """
    An optimized VectorSpace that pre-builds lookup tables for specific dimensions.
    Trades Memory for Speed.
    """

    def __init__(self, allowed_vectors, dim: int, indexed_axes: tuple = (0,), precision: int = 6):
        # 1. Initialize the Parent (VectorStateSpace)
        # This handles the matrix creation and 'dim' assignment
        super().__init__(allowed_vectors, dim=dim)

        # 2. Store indexing parameters
        self.indexed_axes = indexed_axes
        self.precision = precision

        # 3. Build the Index (The "Database")
        self._index_map = self._build_index()

    def _build_index(self) -> dict:
        """Internal helper to construct the hash map."""
        lookup = {axis: defaultdict(list) for axis in self.indexed_axes}

        # We iterate once through the vectors we just stored
        for i, vector in enumerate(self.allowed_vectors):
            vals = vector.values
            for axis in self.indexed_axes:
                if axis < len(vals):
                    # Rounding is crucial for float equality in Hash Maps
                    key = round(vals[axis], self.precision)
                    lookup[axis][key].append(i)
        return lookup

    def search_by_index(self, axis_idx: int, value: float) -> 'VectorStateSpace':
        """O(1) Search using the hash map."""
        # Check if we have an optimized index for this axis
        if axis_idx in self._index_map:
            key = round(value, self.precision)
            # Fast Lookup
            target_indices = self._index_map[axis_idx].get(key, [])

            if not target_indices:
                return self._create_empty()

            # Retrieve vectors directly
            subset_vectors = [self.allowed_vectors[i] for i in target_indices]
            return VectorStateSpace(subset_vectors, dim=self.dim)

        # Fallback to standard O(N) scan if axis is not indexed
        return super().filter_by_index(axis_idx, value, atol=10 ** -self.precision)

    def select_index(self, axes: list[int], values: list[float]) -> 'VectorStateSpace':
        """Compound Search (AND Query)."""
        if len(axes) != len(values):
            raise ValueError("Axes list and Values list must have the same length.")

        candidate_indices = None
        manual_constraints = []

        # 1. Fast Pass (Intersection)
        for axis, val in zip(axes, values):
            if axis in self._index_map:
                key = round(val, self.precision)
                found = set(self._index_map[axis].get(key, []))

                if candidate_indices is None:
                    candidate_indices = found
                else:
                    candidate_indices.intersection_update(found)

                if not candidate_indices:
                    return self._create_empty()
            else:
                manual_constraints.append((axis, val))

        # 2. Handle results
        if candidate_indices is None:
            # If no indexed axes were used, we must check everything (slow path)
            candidate_indices = range(len(self.allowed_vectors))

        # 3. Final Verification (Manual Checks on survivors)
        final_vectors = []
        for idx in candidate_indices:
            vector = self.allowed_vectors[idx]
            match = True
            for axis, val in manual_constraints:
                if not jnp.isclose(vector.values[axis], val, atol=10 ** -self.precision):
                    match = False
                    break
            if match:
                final_vectors.append(vector)

        return VectorStateSpace(final_vectors, dim=self.dim)

    def _create_empty(self):
        return VectorStateSpace([], dim=self.dim)