import jax
import jax.numpy as jnp
from typing import Any, Callable, Union, List

from .base import IOperator, InteractionContext, Observation
from src.field_dynamic_system.core.field.mappings import FieldMapper

SelectionStrategy = Callable[[jnp.ndarray, InteractionContext], Union[int, List[int]]]


class Strategies:
    @staticmethod
    def argmax(buffer: jnp.ndarray, context: InteractionContext) -> int:
        # ravel() ensures we work on a flat array, even if buffer is (N, 1)
        return int(jnp.argmax(buffer.ravel()))

    @staticmethod
    def argmax_all(buffer: jnp.ndarray, context: InteractionContext) -> List[int]:
        flat = buffer.ravel()
        max_val = jnp.max(flat)
        indices = jnp.where(jnp.isclose(flat, max_val))[0]
        return [int(i) for i in indices]

    @staticmethod
    def sample(buffer: jnp.ndarray, context: InteractionContext) -> int:
        if context.rng_key is None:
            raise ValueError("Strategy 'sample' requires rng_key")

        flat = buffer.ravel()
        logits = jnp.log(flat + 1e-10)
        _, subkey = jax.random.split(context.rng_key)

        # Returns a scalar JAX array, cast to int
        idx = jax.random.categorical(subkey, logits)
        return int(idx)


class FieldBasedOperator(IOperator):
    def __init__(self, selection_strategy: SelectionStrategy = Strategies.argmax):
        self.strategy = selection_strategy

    def observe(self, field: FieldMapper, context: InteractionContext) -> Observation:
        raw_buffer = field.raw_buffer
        result_indices = self.strategy(raw_buffer, context)

        if isinstance(result_indices, list):
            return [field.state_space.get_state_by_id(idx) for idx in result_indices]
        else:
            return field.state_space.get_state_by_id(result_indices)