from typing import Any, Callable, Optional
from .base import IOperator, InteractionContext, Observation


class ClassicalOperator(IOperator):
    """
    A Pure Classical Operator.

    It does NOT mutate the system. It strictly transforms input -> output.

    Modes:
    1. Active (Simulator):  Input State + Context -> Next State (Trajectory)
    2. Passive (Viewer):    Input State -> Input State (Identity)
    """

    def __init__(self, transition_fn: Optional[Callable[[Any, int], Any]] = None):
        """
        Args:
            transition_fn: Function f(state, action_id) -> next_state.
                           If None, operator acts as a pass-through (Identity).
        """
        self.transition_fn = transition_fn

    def observe(self, state: Any, context: InteractionContext) -> Observation:
        """
        Args:
            state: The current immutable state (e.g., VectorState).
            context: The input/context for the step.

        Returns:
            The Observed State (either the same state or the next calculated state).
        """
        # 1. Active Mode: Calculate Trajectory
        if self.transition_fn is not None:
            return self.transition_fn(state, context.action_id)

        # 2. Passive Mode: What you see is what you have
        return state