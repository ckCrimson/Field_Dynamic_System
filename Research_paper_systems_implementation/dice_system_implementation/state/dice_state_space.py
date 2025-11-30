import copy

from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from fds.core.fds_state.state_space import ContinuousFiniteStateSpace, S


class DiceStateSpace(ContinuousFiniteStateSpace):

    def __init__(self, currentState: DiceState, dice_number: int = 6,lower_limit: float = None, upper_limit: float = None):

        super().__init__(currentState)
        self.dice_number:int = dice_number
        if currentState is not None and (currentState.state > dice_number or currentState.state < 1):
            raise ValueError(f"currentState ({currentState}) must be between 1 and {self.dice_number}")
        self.state = currentState
        if lower_limit is None or lower_limit <1:
            lower_limit=1
        if upper_limit is None:
            upper_limit=self.dice_number
        self.build_state_space(lower_limit,upper_limit)


    def build_state_space(self, lower_limit:int,upper_limit:int, **kwargs):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def contains(self, s: S) -> bool:
        return s.state>=self.lower_limit and s.state<=self.upper_limit

    def  size(self) -> int:
        return self.dice_number-1

    def get_all_states(self) -> tuple:
        return (self.lower_limit, self.upper_limit)

    def get_state(self) -> DiceState:
        return self.state

    def union_state_space(self, other: 'DiceStateSpace') -> 'DiceStateSpace':
        return DiceStateSpace(self.state,self.dice_number,min(self.lower_limit, other.lower_limit), max(self.upper_limit, other.upper_limit))

    def intersection_state_space(self,  other: 'DiceStateSpace') -> 'DiceStateSpace':
        other_lower = other.lower_limit
        other_upper = other.upper_limit
        if other_lower>self.upper_limit:
            return None
        elif self.lower_limit<other_lower and self.upper_limit<other_upper:
            return copy.deepcopy(other)
        elif self.lower_limit>other_lower and self.upper_limit<other_upper:
            return copy.deepcopy(self)
        else:
            return DiceStateSpace(self.state, self.dice_number, max(self.lower_limit, other.lower_limit),
                                  min(self.upper_limit, self.upper_limit))


    def build_from_initial_state(self, cls, state: DiceState):
        return copy.deepcopy(self)

    def set_state(self, state: DiceState) -> None:
        self.state = state

    def __repr__(self):
        return f"DiceStateSpace({self.state}, {self.dice_number}): [{self.lower_limit},{self.upper_limit}]"

    def __str__(self):
        return f"DiceStateSpace({self.state}, {self.dice_number}): [{self.lower_limit},{self.upper_limit}]"

    def __eq__(self, other):
        if isinstance(other, DiceStateSpace):
            return (self.state == other.state and self.dice_number == other.dice_number and self.lower_limit == other.lower_limit and self.upper_limit == other.upper_limit)








