import copy
from typing import List, Dict

from fds import State
from fds.affecting.affecting_systems_framework.affecting_state_space_mapping import AffectedSystemsMapping
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState


class  CorrerlatedAffectingRWMapping(AffectedSystemsMapping):

    def __init__(self):
        super().__init__()

    def get_affected_state(self, input_system_states: CorrelatedState) -> CorrelatedState:
        """Pack per-system states into one affected/global state."""
        return copy.deepcopy(input_system_states)
    # ----- inverse (affected -> systems) -----

    def get_system_state(self, input_affected_state: CorrelatedState) -> List[Dict[str, State]]:
        """Unpack an affected/global state back to per-system states."""
        dict_systems_state: Dict[str, State] = {}
        for item in input_affected_state.items():
            sys_id, system_state = item
            dict_systems_state[sys_id] = system_state
        return_set = []
        return_set.append(dict_systems_state)
        return return_set