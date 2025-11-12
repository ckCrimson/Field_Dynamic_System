from typing import Optional

from fds.core.fds_state import Reachable, IsAllowedState
from one_dim_random_walker.core.states.state import IntegerState
from one_dim_random_walker.core.states.state_space import IntegerLine


class OneDimensionAllowedState(IsAllowedState):
    def __init__(self, distance: int  = 1):
        self.distance = distance
        if self.distance < 0:
            raise ValueError("The distance must be greater than equal zero.")
        super().__init__()

    def is_state_allowed(self, state_initial: IntegerState, state: IntegerState) -> bool:
        if self.distance>0:
          return abs(( state_initial - state ).state) <= self.distance
        return False


class OneDimensionReachable(Reachable):
    def __init__(self, distance: int = 1):
        self.distance = distance
        if self.distance < 0:
            raise ValueError("The distance must be greater than equal zero.")
        is_allowed =  OneDimensionAllowedState(self.distance)
        super().__init__(is_allowed = is_allowed)

    def  get_reachable(
        self,
        state: IntegerState,
        within_state_space: bool = True,
        from_allowed: Optional[bool] = False,
        state_space: Optional[IntegerLine] = None,
    ) -> IntegerLine:
        return IntegerLine(current_state= state,
        left_limit= state.state - self.distance,
        right_limit= state.state+self.distance)

