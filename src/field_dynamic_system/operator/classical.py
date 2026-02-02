from typing import Any, Callable, Optional
from .base import IOperator, InteractionContext, Observation


class ClassicalOperator(IOperator):
    """
    A Dual-Mode Classical Observer.

    1. Active Mode: If initialized with a transition_fn, it drives the physics.
    2. Passive Mode: If initialized with None, it simply observes.
    """

    def __init__(self, transition_fn: Optional[Callable[[Any, int], Any]] = None):
        """
        Args:
            transition_fn: Optional function f(state, action) -> new_state.
                           If None, the operator acts as a read-only viewer.
        """
        self.transition_fn = transition_fn

    def observe(self, system_state: Any, context: InteractionContext) -> Observation:
        # 1. READ
        current_val = system_state.value

        # 2. LOGIC (Conditional)
        if self.transition_fn is not None:
            # Active Mode: Calculate -> Update
            next_val = self.transition_fn(current_val, context.action_id)
            system_state.value = next_val
            return next_val

        # 3. RETURN (Passive Mode)
        # No logic provided, so we just return what we see.
        return current_val