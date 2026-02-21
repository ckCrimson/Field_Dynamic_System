import jax.numpy as jnp
from typing import Any
from .base import IOperator, InteractionContext
from src.field_dynamic_system.core.field.mappings import FieldMapper


class ExpectationOperator(IOperator):
    """
    Calculates E[X] over the provided Field.
    """

    def observe(self, field: FieldMapper, context: InteractionContext) -> float:
        # FIX: Flatten (N, 1) -> (N,) to align with values_array shape
        probs = field.raw_buffer.ravel()

        # 1. Extract Values
        values = []
        for i in range(field.state_space.num_states):
            state = field.state_space.get_state_by_id(i)
            # Support both objects with .value and raw numbers
            val = getattr(state, 'value', state)
            values.append(float(val))

        values_array = jnp.array(values)

        # 2. Compute Expectation
        # Now shapes match: (N,) dot (N,) -> scalar
        return float(jnp.dot(probs, values_array))

    def selection_strategy(self) -> Any:
        return None