from collections.abc import Iterable

from fds.affecting.symmetric_systems_framework.pivot_state import PivotState
from fds.core.fds_state.state_space import  DiscreteFiniteStatSpace


class SymmetricStateSpace(DiscreteFiniteStatSpace):
    def __init__(self,states: Iterable[ PivotState ], current_state:PivotState):
        super().__init__(states, current_state)