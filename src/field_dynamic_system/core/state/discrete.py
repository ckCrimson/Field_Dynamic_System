import jax.numpy as jnp
from .interfaces import IDiscreteStateSpace, StateEncoder, StateSpace, State
from .encoding import BitMaskingEncoding, VectorEncoding
from .state import VectorState, AbstractState

from typing import Set, List, Union, Sequence, Any


class AbstractDiscreteStateSpace(IDiscreteStateSpace):
    """
    A Finite Set of AbstractStates (e.g. {Rock, Paper, Scissor}).
    Uses Set theory for fast lookups.
    """

    def __init__(self, states: Union[Set[AbstractState], Sequence[AbstractState]]):
        # Convert to set to remove duplicates and enable O(1) lookups
        self.allowed_states = set(states)

        # We sort the list for deterministic encoding (Rock is always 0, Paper always 1)
        # This prevents random ID swapping between runs.
        # Note: AbstractState must support sorting (e.g., by name) or we just use list(set).
        sorted_states = sorted(list(self.allowed_states), key=lambda x: str(x))

        # Auto-initialize the encoder with the known set
        self._encoder = BitMaskingEncoding(sorted_states)
        self._num_states = len(sorted_states)

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


class VectorStateSpace(IDiscreteStateSpace):
    """
    A Finite Set of specific VectorStates.
    Example: { Up=(0,1), Down=(0,-1) }
    """

    def __init__(self, vectors: List[VectorState], dim: int):
        # We convert to a set immediately to remove duplicates
        self.allowed_vectors = tuple(set(vectors))
        self._dim = dim
        self._encoder = VectorEncoding(dim)

        # Pre-compute the matrix of valid vectors for fast JAX lookup
        # Shape: (M, D) where M is number of allowed vectors
        if self.allowed_vectors:
            raw_list = [v.values for v in self.allowed_vectors]
            self._matrix = jnp.array(raw_list, dtype=jnp.float32)
        else:
            # Handle empty space case
            self._matrix = jnp.zeros((0, dim), dtype=jnp.float32)

    @property
    def encoder(self) -> StateEncoder:
        return self._encoder

    def get_matrix(self) -> jnp.ndarray:
        return self._matrix

    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        # ... (Same implementation as previous step) ...
        if not isinstance(state, jnp.ndarray):
            vecs = self.encoder.encode(state)
        else:
            vecs = state

        if vecs.ndim == 1:
            vecs = vecs[None, :]

        # If space is empty, nothing is contained
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
        """
        Combines allowed vectors from both spaces.
        Example: {Up} U {Down} = {Up, Down}
        """
        if isinstance(other, VectorStateSpace):
            # Check dimension compatibility
            if self._dim != other._dim:
                raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")

            # Set Union logic (Fast because VectorState is hashable)
            new_set = set(self.allowed_vectors).union(set(other.allowed_vectors))
            return VectorStateSpace(list(new_set), self._dim)

        raise TypeError(f"Cannot union VectorStateSpace with {type(other)}")

    def intersection(self, other: StateSpace) -> StateSpace:
        """
        Keeps only vectors present in BOTH spaces.
        Example: {Up, Down} ∩ {Up, Left} = {Up}
        """
        if isinstance(other, VectorStateSpace):
            if self._dim != other._dim:
                raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")

            # Set Intersection logic
            new_set = set(self.allowed_vectors).intersection(set(other.allowed_vectors))
            return VectorStateSpace(list(new_set), self._dim)

        raise TypeError(f"Cannot intersect VectorStateSpace with {type(other)}")