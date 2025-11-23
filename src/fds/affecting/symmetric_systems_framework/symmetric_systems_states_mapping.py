from typing import Dict


from fds import State, StatSpace
from fds.affecting.affecting_systems_framework.affecting_state_space_mapping import AffectedSystemsMapping
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.core.fds_state import StateSpaceMapping


class SymmetricStateSpaceMapping(AffectedSystemsMapping):
    def __init__(self, pivot_system_name: str, mapping_list: Dict[str, StateSpaceMapping], proto_state_space: Dict[str,StatSpace]):
        self.pivot_system_name = pivot_system_name
        self.mapping_list = mapping_list
        self.proto_state_space = proto_state_space
        super().__init__()

    def get_affected_state(self, input_system_states: CorrelatedState) -> State:
        """This returns the state of the pivot system from the input system states"""
        pivot_system_state = input_system_states.get_component(self.pivot_system_name)
        return pivot_system_state

    def get_system_state(self, input_affected_state: State) -> CorrelatedState:
        """input_affected_state : the input pivot state is used to find mapping of the pivot state onto each state which would return the
         correlatesd state
        """
        dictionry_of_states : Dict[str,State] = {}
        for system_name , mappping in self.mapping_list.items():
            mapped_state_space = mappping.get_mapping(input_affected_state,self.proto_state_space[system_name])
            new_system_state = mapped_state_space.get_state()
            dictionry_of_states[system_name] = new_system_state
        return CorrelatedState(dictionry_of_states)