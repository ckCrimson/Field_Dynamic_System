import copy
from typing import List, Dict

from fds import State
from fds.affecting.affecting_systems_framework.affecting_state_space_mapping import AffectedSystemsMapping
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace


class SystemCorrelatedSpaceMapping(AffectedSystemsMapping):
    def __init__(self):
        super().__init__()

    def get_affected_state(self, correlated_state: CorrelatedState) -> CorrelatedState:
        return copy.deepcopy( correlated_state )

    def get_system_state(self, correlated_state: CorrelatedState) -> List[Dict[str,State]]:
        dict_systems_state :Dict[str,State] = {}
        print(correlated_state)
        for item in correlated_state.items():
            sys_id, system_state = item
            dict_systems_state[sys_id]=system_state
        return_set= []
        return_set.append(dict_systems_state)
        return return_set

    def get_mapping(self,correlated_input_state: CorrelatedState, output_state_space: CorrelatedStateSpace) -> CorrelatedStateSpace:
        new_ouput_state_space = copy.deepcopy(output_state_space)
        return new_ouput_state_space.build_from_states({correlated_input_state})

