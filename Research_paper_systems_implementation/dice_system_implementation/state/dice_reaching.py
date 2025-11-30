from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds import StatSpace
from fds.core.fds_state import Reaching
from fds.core.fds_state.reaching import S


class DiceReaching(Reaching):
    def __init__(self,alpha: int,n_o:int,d:int):
        self.alpha = alpha
        self.n_o = n_o
        self.d = d
        super().__init__()

    def get_reaching(self, state: DiceState) -> DiceStateSpace:
        total_throws = self.alpha + self.n_o
        lower_limit = (state.state * total_throws) -  (self.d * self.alpha)
        upper_limit = (state.state * total_throws) - (self.alpha)
        return DiceStateSpace(state,self.d,lower_limit/self.n_o, upper_limit/self.n_o)