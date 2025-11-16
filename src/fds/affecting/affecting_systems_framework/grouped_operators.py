from abc import ABC
from typing import Dict, List

from fds import State, Field
from fds.affecting.affecting_systems_framework.affecting_state_space_mapping import AffectedSystemsMapping
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.dynamics.fds_operator.operators import Operator


class AffectedSystemsOperator(Operator, ABC):
    def __init__(self, affecting_group:  Dict[str, FieldDynamicSystem], mapping: AffectedSystemsMapping ,*args,**kwargs):
        super().__init__()
        self.affecting_group = affecting_group
        self.mapping = mapping

    def pick_state_from_set(self,state_set: List):
        """A policy to pick state from the set of states for system level mapping. If not overriden picks any random state"""
        if len(state_set)>0:
            return state_set[0]




    def apply(self,field: Field,history: list[State]):
        next_state = self.get_next_state(field)
        history.append(next_state)
        system_states_set = self.mapping.get_system_state(next_state)
        system_states = self.pick_state_from_set(system_states_set)
        for system_id,systems in self.affecting_group.items():
            system_state = system_states[system_id]
            self.affecting_group[system_id].initial_state = system_state
            self.affecting_group[system_id].transition_list.append(system_state)


