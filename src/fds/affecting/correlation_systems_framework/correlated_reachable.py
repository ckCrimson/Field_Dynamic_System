from abc import ABC
from typing import Dict, cast

from fds import State
from fds.affecting.affecting_systems_framework.affected_reachable import AffectedReachable
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem


class CorrelatedAffectedReachable(AffectedReachable, ABC):
    def __init__(self,members: Dict[str, FieldDynamicSystem]):
        super().__init__(members)

    def get_reachable(self, state: CorrelatedState = None,within_state_space =False,from_allowed:bool=False) -> CorrelatedStateSpace:
        if state is None:
            dictionary_of_correlated_state : Dict[str,State] = {}
            for sys_id, sys in self.members.items():
                dictionary_of_correlated_state[sys_id] = sys.initial_state
            state = CorrelatedState(dictionary_of_correlated_state)
        return cast(CorrelatedStateSpace ,self.get_affected_reachable(state))
