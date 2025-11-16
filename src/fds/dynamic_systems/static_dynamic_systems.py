from abc import ABC
from typing import TypeVar, Generic

from fds import State, StatSpace
from fds.core.fds_state import Reachable, Reaching
from fds.dynamics.multi_step import MultiStepReaching

T = TypeVar("T", bound=State)
class StaticDynamicSystem(ABC, Generic[T]):
    """
    System without evolution: has state space, reachable, and reaching.
    """
    def __init__(
        self,
        initial_state: State,
        state_space: StatSpace[T],
        reachable: Reachable[T],
        reaching: Reaching[T]=None,
        multi_step_reaching: MultiStepReaching[T]=None
            # you can keep this if you have separate reverse logic
    ):
        self.initial_state = initial_state
        self.state_space = state_space
        self.reachable = reachable
        self.reaching = reaching
        self.multi_step_reaching = multi_step_reaching
