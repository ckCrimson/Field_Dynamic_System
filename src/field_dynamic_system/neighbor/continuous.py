from typing import Any

from .interfaces import Topology
from ..core.state.interfaces import IContinuousStateSpace, StateSpace

# We need a way to represent the "Result" of a successor query.
# Usually, this is a basic geometric shape.
from ..core.state.continous import HypersphereSpace  # Assuming this exists or we create it

from abc import abstractmethod



class ContinuousTopology(Topology):
    """
    Base class for Continuous Reachability.

    Unlike DiscreteTopology, we CANNOT automatically compile a 'matrix'
    for multi-step reachability because the state space is uncountable.

    Strategy:
    - Simple Topologies (Metric): We implement exact multi-step.
    - Complex Topologies: The User must implement multi-step.
    """

    def __init__(self, state_space: IContinuousStateSpace):
        if not isinstance(state_space, IContinuousStateSpace):
            raise TypeError("ContinuousTopology requires an IContinuousStateSpace.")
        super().__init__(state_space)

    @abstractmethod
    def successor(self, state: Any) -> StateSpace:
        """Returns the immediate reachable set (One Step)."""
        pass

    def multi_step_successor(self, initial_state: Any, steps: int) -> StateSpace:
        """
        Calculates reachability after N steps.

        Default Implementation:
        If the user does not override this, we try to perform recursive expansion
        assuming the StateSpace object supports some form of 'minkowski_sum' or 'expansion'.

        If that is impossible, we raise an Error forcing the user to implement it.
        """
        # 1. Base Case
        if steps == 0:
            # We need to wrap the single point into a Shape (e.g., 0-radius ball)
            # This depends on the space type. Let's assume HyperSphere for generic R^n.
            return HypersphereSpace(initial_state, radius=0.0)

        # 2. Optimization Hook
        # Users should override this method for exact math (e.g. radius * steps).
        return self._compute_multi_step_manually(initial_state, steps)

    def _compute_multi_step_manually(self, initial_state: Any, steps: int) -> StateSpace:
        """
        Fallback logic.
        For many continuous systems, R^k(x) is difficult to compute generically.
        """
        raise NotImplementedError(
            "Exact multi-step reachability cannot be computed generically for continuous spaces. "
            "Please implement 'multi_step_successor' in your Topology subclass "
            "or use a specific implementation like MetricTopology."
        )


# --- CONCRETE IMPLEMENTATION 1: Metric Topology ---
# This is the "Standard" one we provide.

class MetricTopology(ContinuousTopology):
    """
    Reachability is defined by Distance.
    R(x) = Ball(x, epsilon)
    """

    def __init__(self, state_space: IContinuousStateSpace, epsilon: float):
        super().__init__(state_space)
        self.epsilon = epsilon

    def successor(self, state: Any) -> StateSpace:
        return HypersphereSpace(center=state, radius=self.epsilon)

    def multi_step_successor(self, initial_state: Any, steps: int) -> StateSpace:
        """
        Exact Implementation:
        The reachable set grows linearly with steps.
        radius_new = radius_old + (steps * epsilon)
        """
        # Starting from a point implies initial radius is 0
        total_radius = steps * self.epsilon
        return HypersphereSpace(center=initial_state, radius=total_radius)