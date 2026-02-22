from typing import Any

from .interfaces import Topology
from ..core.state.interfaces import IContinuousStateSpace, StateSpace

# We need a way to represent the "Result" of a successor query.
# Usually, this is a basic geometric shape.
from ..core.state.continous import HypersphereSpace  # Assuming this exists or we create it

from abc import abstractmethod

from abc import abstractmethod
from typing import Any


# Assuming imports: Topology, IContinuousStateSpace, StateSpace, HypersphereSpace

class ContinuousTopology(Topology):
    """
    Base class for Continuous Reachability.
    Supports both OOP StateSpace generation and raw DOD boundary calculations.
    """

    def __init__(self, state_space: 'IContinuousStateSpace'):
        if not isinstance(state_space, IContinuousStateSpace):
            raise TypeError("ContinuousTopology requires an IContinuousStateSpace.")
        super().__init__(state_space)

    # --- OOP PATH (Human/Orchestrator Facing) ---

    @abstractmethod
    def successor(self, state: Any) -> 'StateSpace':
        """Returns the immediate reachable set (One Step) as an OOP StateSpace object."""
        pass

    def multi_step_successor(self, initial_state: Any, steps: int) -> 'StateSpace':
        """Calculates reachability after N steps as an OOP StateSpace object."""
        if steps == 0:
            return HypersphereSpace(initial_state, radius=0.0)
        return self._compute_multi_step_manually(initial_state, steps)

    def _compute_multi_step_manually(self, initial_state: Any, steps: int) -> 'StateSpace':
        raise NotImplementedError(
            "Exact multi-step reachability cannot be computed generically for continuous spaces. "
            "Please implement 'multi_step_successor'."
        )

    # --- RAW DOD PATH (JAX/GPU Facing) ---

    
    def get_raw_successor(self, state_raw: Any) -> Any:
        """
        Returns the immediate reachable bounds as raw arrays.
        For JAX performance, this MUST return flat arrays or tuples of arrays.
        Example: Returns (low_array, high_array) for a bounding box.
        """
        pass

    def get_raw_multi_step_successor(self, initial_state_raw: Any, steps: int) -> Any:
        """
        Returns the reachable bounds after N steps as raw arrays.
        """
        if steps == 0:
            # Base case: The bounds are just the exact starting point.
            # (Assuming the raw format expects a tuple of (low, high) bounds)
            return (initial_state_raw, initial_state_raw)

        return self._compute_raw_multi_step_manually(initial_state_raw, steps)

    def _compute_raw_multi_step_manually(self, initial_state_raw: Any, steps: int) -> Any:
        """
        Fallback logic for raw multi-step arrays.
        """
        raise NotImplementedError(
            "Raw multi-step reachability must be implemented by the specific ContinuousTopology subclass. "
            "It should return mathematically expanded bounds (e.g., Minkowski sum) as raw arrays."
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