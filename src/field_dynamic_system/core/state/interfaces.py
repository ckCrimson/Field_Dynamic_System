from abc import abstractmethod
from typing import Protocol, Any, runtime_checkable, List

import jax.tree_util

# --- 1. Base State Marker ---
@runtime_checkable
class State(Protocol):
    """Marker interface for any state object (Vector or Abstract)."""
    pass


# --- 2. Encoder Interface ---
from typing import Protocol, Union, Sequence, runtime_checkable, Any
import jax.numpy as jnp


# ... State protocol remains same ...

@runtime_checkable
class StateEncoder(Protocol):
    """
    Responsible for translating high-level State objects into JAX arrays.
    Handles both single items and batches dynamically.
    """

    def encode(self, data: Union['State', Sequence['State']]) -> jnp.ndarray:
        """
        Encodes a single state OR a list of states.
        Automatic dispatch to optimized batching if a list is provided.
        """
        ...

    def decode(self, encoded_data: jnp.ndarray) -> 'State':
        ...

    def __eq__(self, other):
        # Value-based equality
        if type(self) != type(other):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):
        # Hash based on attributes (assumes attributes are hashable)
        # Convert dicts to frozensets for hashing if needed
        return hash(tuple(sorted(self.__dict__.items())))

    @property
    def shape(self) -> tuple[int, ...]:
        ...
# --- 3. State Space Hierarchy ---
@runtime_checkable
class StateSpace(Protocol):
    """
    Base contract for all Spaces.
    """

    # Updated signature: accepts List, returns Array
    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """
        Checks validity.
        - Input: Single State -> Returns bool
        - Input: List[State] -> Returns bool array (Batch)
        - Input: JAX Array   -> Returns bool array (Batch/Single based on shape)
        """

    def union(self, other: 'StateSpace') -> 'StateSpace':
        """Logic Operation: OR"""
        ...

    def intersection(self, other: 'StateSpace') -> 'StateSpace':
        """Logic Operation: AND"""
        ...


@runtime_checkable
class IContinuousStateSpace(StateSpace, Protocol):
    """
    For Geometry-based spaces (Hypercubes, Spheres).
    """

    # --- 1. The Helper: Standardizes Input ---
    def _resolve_to_array(self, state: Union[State, Sequence[State], jnp.ndarray]) -> jnp.ndarray:
        """
        Converts generic input (List, Object, or Array) into a JAX array
        using the space's encoder.
        """
        # Case A: Already a JAX Array (Fastest, used internally)
        if isinstance(state, jnp.ndarray):
            return state

        # Case B: Python Object(s) -> Use our smart Encoder
        return self.encoding.encode(state)

    # --- 2. Abstract Logic ---
    @abstractmethod
    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        ...

    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        """Math Enforcement: Snap invalid values to the manifold."""
        ...

@runtime_checkable
class IStateOperation(Protocol):
    """
    Defines an operation to be performed on a State.
    """
    def __call__(self, state: State) -> Any:
        """
        The logic to apply.
        NOTE: For VectorStates, 'state' might be a Batched Object
        containing arrays instead of scalars. Write JAX-compatible math!
        """
        ...


@runtime_checkable
class IDiscreteStateSpace(StateSpace, Protocol):
    """
    For Set-based spaces (Finite options).
    """

    @property
    def encoder(self) -> StateEncoder:
        """Reference to the specific encoding scheme used."""
        ...

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Returns the total count of states in this space."""
        pass

    def get_matrix(self) -> jnp.ndarray:
        """
        Returns a JAX array containing ALL valid encoded states.
        Used for iterating over the space inside the physics kernel.
        """
        ...

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
            # shape check for batch vs single handled by JAX broadcasting
            return (state >= 0) & (state < self._num_states)

        # Case B: Python List (Batch)
        if isinstance(state, list) or isinstance(state, tuple):
            # Python Set lookup is faster than JAX for abstract objects
            # Returns boolean array
            mask = [s in self.allowed_states for s in state]
            return jnp.array(mask, dtype=bool)

        # Case C: Single Object
        return state in self.allowed_states

    def map(self, operation: IStateOperation):
        pass

    def __init_subclass__(cls, **kwargs):
        """
        GENERIC IMPLEMENTATION:
        Automatically registers any subclass as a JAX Pytree.
        """
        super().__init_subclass__(**kwargs)

        # We tell JAX: "When you see this class, call these specific methods"
        jax.tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten
        )

    @abstractmethod
    def _tree_flatten(self):
        """
        SPECIALIZED: Must return (children, aux_data).
        - Children: JAX Arrays (Dynamic/Differentiable).
        - Aux: Static Metadata (Integers, Strings, Tuples).
        """
        pass

    @classmethod
    @abstractmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        SPECIALIZED: Must reconstruct the object from the data.
        """
        pass

    @abstractmethod
    def get_index_of(self, state_obj: Any) -> int:
        """
        Returns the integer index (0 to N-1) of the given state object.
        Returns -1 if the state is not found in this space.
        """
        pass


    @property
    @abstractmethod
    def states(self) -> List[Any]:
        """Returns the raw Python list of state objects."""
        pass


    @abstractmethod
    def create_subset(self, states: List[Any]) -> 'IDiscreteStateSpace':
        """
        Factory method: Creates a NEW Discrete State Space containing
        only the provided list of states.
        """
        pass

@runtime_checkable
class IStateSpaceTransformation(Protocol):
    """
    The Contract: Takes a StateSpace, returns a new StateSpace.
    """
    def transform(self, space: StateSpace) -> StateSpace:
        pass