from dataclasses import dataclass

import jax.numpy as jnp
from .interfaces import StateEncoder, State
from .state import VectorState, AbstractState
from typing import List, Union, Sequence, Tuple, Any


# class VectorEncoding(StateEncoder):
#     def __init__(self, dim: int):
#         self.dim = dim
#
#     def encode(self, data: Union[State, Sequence[State]]) -> jnp.ndarray:
#         # 1. BATCH PATH (Fastest for lists)
#         if isinstance(data, list) or isinstance(data, tuple):
#             # Optimistic check: Assume all elements are VectorState for speed
#             # Extract tuples: [(1.0, 2.0), (3.0, 4.0)]
#             raw_values = [s.values for s in data]
#             return jnp.array(raw_values, dtype=jnp.float32)
#
#         # 2. SINGLE PATH
#         if isinstance(data, VectorState):
#             return jnp.array(data.values, dtype=jnp.float32)
#
#         raise TypeError(f"Expected VectorState or List[VectorState], got {type(data)}")
#
#     def decode(self, data: jnp.ndarray) -> VectorState:
#         return VectorState(tuple(data.tolist()))
#
#     @property
#     def shape(self) -> tuple[int, ...]:
#         return (self.dim,)

@dataclass
class VectorEncoding(StateEncoder):
    dim: int

    def encode(self, data: Union[State, Sequence[State]]) -> jnp.ndarray:
        if isinstance(data, VectorState):
            return jnp.array(data.values)
        # Handle list of states
        return jnp.array([s.values for s in data])

    def decode(self, encoded_data: jnp.ndarray) -> State:
        return VectorState(tuple(encoded_data.tolist()))

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.dim,)


class BitMaskingEncoding(StateEncoder):
    def __init__(self, states: List[AbstractState]):
        self.obj_to_id = {s: i for i, s in enumerate(states)}
        self.id_to_obj = {i: s for i, s in enumerate(states)}

    def encode(self, data: Union[State, Sequence[State]]) -> jnp.ndarray:
        # 1. BATCH PATH
        if isinstance(data, list) or isinstance(data, tuple):
            # Extract IDs: [0, 5, 2...]
            raw_ids = [self.obj_to_id[s] for s in data]
            # Reshape is crucial for JAX grids: (N, 1)
            return jnp.array(raw_ids, dtype=jnp.int32).reshape(-1, 1)

        # 2. SINGLE PATH
        # Assume AbstractState hashable check passes via dictionary lookup
        try:
            return jnp.array([self.obj_to_id[data]], dtype=jnp.int32)
        except KeyError:
            raise TypeError(f"Object {data} not found in Discrete Registry or invalid type.")

    def decode(self, data: jnp.ndarray) -> State:
        idx = int(data.item())
        return self.id_to_obj[idx]

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)


# ... existing StateEncoder base class ...

class IdentityEncoder(StateEncoder):
    """
    Pass-through encoder for Continuous Spaces.
    Input: Vector R^n
    Output: Vector R^n (Unchanged)
    """
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def output_dim(self) -> int:
        return self._dim

    def encode(self, state: Union[Any, jnp.ndarray]) -> jnp.ndarray:
        """
        Expects input to already be numeric (float/int).
        Just ensures it is a JAX array of correct shape.
        """
        # 1. Convert to JAX Array
        if not isinstance(state, jnp.ndarray):
            # Handle object wrappers like VectorState
            if hasattr(state, 'values'):
                arr = jnp.array(state.values, dtype=jnp.float32)
            else:
                arr = jnp.array(state, dtype=jnp.float32)
        else:
            arr = state

        # 2. Validation (Optional, can be skipped for raw speed)
        # Checks if the last dimension matches self._dim
        # if arr.shape[-1] != self._dim:
        #     raise ValueError(f"Expected dim {self._dim}, got {arr.shape[-1]}")

        return arr.astype(jnp.float32)

    def decode(self, encoded_state: jnp.ndarray) -> Any:
        """
        Identity: Returns the vector as-is.
        """
        return encoded_state
