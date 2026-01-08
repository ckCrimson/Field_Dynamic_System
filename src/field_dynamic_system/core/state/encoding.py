import jax.numpy as jnp
from .interfaces import StateEncoder, State
from .state import VectorState, AbstractState
from typing import List, Union, Sequence


class VectorEncoding(StateEncoder):
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, data: Union[State, Sequence[State]]) -> jnp.ndarray:
        # 1. BATCH PATH (Fastest for lists)
        if isinstance(data, list) or isinstance(data, tuple):
            # Optimistic check: Assume all elements are VectorState for speed
            # Extract tuples: [(1.0, 2.0), (3.0, 4.0)]
            raw_values = [s.values for s in data]
            return jnp.array(raw_values, dtype=jnp.float32)

        # 2. SINGLE PATH
        if isinstance(data, VectorState):
            return jnp.array(data.values, dtype=jnp.float32)

        raise TypeError(f"Expected VectorState or List[VectorState], got {type(data)}")

    def decode(self, data: jnp.ndarray) -> VectorState:
        return VectorState(tuple(data.tolist()))

    @property
    def shape(self) -> tuple[int, ...]:
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