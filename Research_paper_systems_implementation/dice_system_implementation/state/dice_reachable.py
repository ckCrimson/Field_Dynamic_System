from typing import Optional

from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds import StatSpace
from fds.core.fds_state import Reachable
from fds.core.fds_state.reachable import S


class DiceReachable(Reachable):
    def __init__(self, alpha:int , n_0: int, d:int):
        """

        alpha: Number of more throws
        n_0: Initial Number of throws
        d : Dice Number

        """
        self.alpha = alpha
        self.n_0 = n_0
        self.d = d
        super().__init__()

    def get_reachable(
        self,
        state: DiceState,
        within_state_space: bool = False,
        from_allowed: Optional[bool] = None,
        state_space: Optional[StatSpace] = None
    ) -> DiceStateSpace:
        total_throws = self.alpha + self.n_0
        lower_limit = (self.alpha + (self.n_0 * state.state))/(total_throws)
        upper_limit = ((self.alpha*self.d) + (self.n_0 * state.state))/(total_throws)
        return DiceStateSpace(state, dice_number =self.d,lower_limit = lower_limit, upper_limit= upper_limit)
