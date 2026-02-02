import jax
import jax.numpy as jnp
from typing import Any, Callable

from .base import IOperator, InteractionContext, Observation

# --- 1. Define the Strategy Signature ---
# A pure function that looks at the field and context, and picks a State.
SelectionStrategy = Callable[['FieldMapper', InteractionContext], Observation]


# --- 2. Standard Strategies (The "Menu") ---
class Strategies:
    """Collection of standard field collapse strategies."""

    @staticmethod
    def argmax(field_mapper: Any, context: InteractionContext) -> Observation:
        """Deterministic: Picks the state with highest probability."""
        idx = jnp.argmax(field_mapper.raw_buffer)
        return field_mapper.index_to_state(int(idx))

    @staticmethod
    def sample(field_mapper: Any, context: InteractionContext) -> Observation:
        """Probabilistic: Samples based on field distribution."""
        if context.rng_key is None:
            raise ValueError("Strategy 'sample' requires rng_key in context")

        logits = jnp.log(field_mapper.raw_buffer + 1e-10)
        _, subkey = jax.random.split(context.rng_key)
        idx = jax.random.categorical(subkey, logits)

        return field_mapper.index_to_state(int(idx))


# --- 3. The Unified Operator ---

class FieldBasedOperator(IOperator):
    """
    A generic observer for Field systems.

    Instead of subclassing, you inject the 'selection_strategy'
    (Physics of Choice) at runtime.
    """

    def __init__(self, selection_strategy: SelectionStrategy = Strategies.argmax):
        """
        Args:
            selection_strategy: Function(field, context) -> state
                                Defaults to Deterministic ArgMax.
        """
        self.strategy = selection_strategy

    def observe(self, system_state: Any, context: InteractionContext) -> Observation:
        # 1. READ
        current_field = system_state.field_mapper

        # 2. SELECT (Delegate to the Strategy Function)
        observation_state = self.strategy(current_field, context)

        # 3. ENCODE & VALIDATE (Safety Check)
        final_state_val = observation_state[-1] if isinstance(observation_state, list) else observation_state

        try:
            final_index = current_field.state_to_index(final_state_val)
        except ValueError as e:
            raise ValueError(f"Strategy selected invalid state {final_state_val}") from e

        # 4. COLLAPSE (The "Measurement" Law)
        current_field.collapse_to_impulse(final_index)

        # 5. RETURN
        return observation_state