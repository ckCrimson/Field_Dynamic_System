from abc import ABC, abstractmethod
from typing import Any, Union
from ..core.state.interfaces import StateSpace

class ITopology(ABC):
    """
    Contract for defining connectivity.
    A Topology maps an initial state (or set of states) to a new StateSpace
    representing the reachable subset.
    """

    @abstractmethod
    def successor(self, initial_state: Any) -> StateSpace:
        """
        V[s]: Returns the subset of the total state space reachable in one step.
        """
        pass

    def predecessor(self, state: Any) -> StateSpace:
        """
        R[s]: Returns the subset of the total state space that can reach
        the input state in one step. (Optional implementation).
        """
        pass

    @abstractmethod
    def multi_step_successor(self, initial_state: Any, steps: int) -> StateSpace:
        """
        L^l[s]: The reachability set after exactly N steps using recursive expansion.
        """
        pass


class Topology(ITopology):
    """
    Base implementation holding the reference to the Global State Space.
    """

    def __init__(self, state_space: StateSpace):
        self._state_space = state_space

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    def predecessor(self, state: Any) -> StateSpace:
        """
        Default safety: Not all topologies are reversible.
        """
        raise NotImplementedError(
            f"Predecessor logic is not defined for {self.__class__.__name__}."
        )

    def multi_step_successor(self, initial_state: Any, steps: int) -> StateSpace:
        """
        Generic expansion logic.
        Note: For Continuous Topologies, this might represent Time Evolution.
        For Discrete Topologies, this represents Iterative Steps.

        We delegate this to the subclass because the algorithm depends entirely
        on whether the space is Countable (Discrete) or Measurable (Continuous).
        """
        raise NotImplementedError(
            "Multi-step logic must be implemented by the specific Discrete/Continuous Topology."
        )