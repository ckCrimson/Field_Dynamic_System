from typing import Any, Callable
import jax.numpy as jnp

from .interfaces import Topology
from ..core.state.interfaces import IContinuousStateSpace, StateSpace

# We need a way to represent the "Result" of a successor query.
# Usually, this is a basic geometric shape.
from ..core.state.continous import HypersphereSpace  # Assuming this exists or we create it


class ContinuousTopology(Topology):
    """
    Topology for Continuous, Uncountable State Spaces (Manifolds).
    """

    def __init__(self, state_space: IContinuousStateSpace):
        if not isinstance(state_space, IContinuousStateSpace):
            raise TypeError("ContinuousTopology requires an IContinuousStateSpace.")
        super().__init__(state_space)
        self.continuous_space = state_space

    def multi_step_successor(self, initial_state: Any, steps: int) -> StateSpace:
        """
        In Continuous terms, 'steps' usually implies Time Steps (dt).
        Recursive application: S_t+1 = Successor(S_t)
        """
        current_region = self.successor(initial_state)

        # For metric topology, step 2 is just a larger ball.
        # For dynamic topology, it is numerical integration.
        # We delegate to the specific logic of the subclass if possible,
        # or chain the successor calls.

        for _ in range(1, steps):
            # This is tricky in continuous: Union of balls around every point in a ball?
            # That is mathematically just a larger ball: Radius = r * steps.
            # We let the subclass handle this optimization.
            current_region = self._expand_region(current_region)

        return current_region

    def _expand_region(self, region: StateSpace) -> StateSpace:
        """Helper for multi-step expansion."""
        raise NotImplementedError("Subclass must define how regions expand over steps.")


class MetricTopology(ContinuousTopology):
    """
    Defines connectivity based on distance (Epsilon-Ball).
    Reachable(x) = { y | dist(x, y) < radius }
    """

    def __init__(self, state_space: IContinuousStateSpace, radius: float, metric_fn: Callable = None):
        super().__init__(state_space)
        self.radius = radius
        # Default to Euclidean if no metric provided
        self.metric_fn = metric_fn if metric_fn else lambda a, b: jnp.linalg.norm(a - b)

    def successor(self, state: Any) -> StateSpace:
        """
        Returns a StateSpace representing the neighborhood Ball.
        """
        # We return a HyperSphere centered at the current state
        # This is a valid 'Subset' of the continuous space.
        return HypersphereSpace(center=state, radius=self.radius)

    def predecessor(self, state: Any) -> StateSpace:
        """
        Symmetric Metric: If I can reach y from x, I can reach x from y.
        """
        return self.successor(state)

    def _expand_region(self, region: StateSpace) -> StateSpace:
        """
        Optimization: A ball of radius R expanded by radius r becomes Ball(R+r).
        """
        if isinstance(region, HypersphereSpace):
            # L^2[x] = Ball(x, 2r)
            return HypersphereSpace(center=region.center, radius=region.radius + self.radius)

        # Fallback for complex shapes (Minkowski Sum approximation)
        # For now, just return the region or raise error
        return region