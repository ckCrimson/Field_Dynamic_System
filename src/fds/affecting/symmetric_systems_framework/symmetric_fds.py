from typing import Dict

from numpy import integer

from fds import StatSpace, Field
from fds.affecting.affecting_systems_framework.affecting_fds import AffectingFDS
from fds.affecting.symmetric_systems_framework.symmetric_systems_states_mapping import SymmetricStateSpaceMapping
from fds.core.fds_state import StateSpaceMapping
from fds.dynamic_systems import FieldDynamicSystem

class SymmetricFDS(AffectingFDS):
    def __init__(self, members: Dict[str, FieldDynamicSystem], pivot_element_name: str,
                 mapping_from_pivot: Dict[str,StateSpaceMapping]):
        self.pivot_element_name = pivot_element_name
        self.mapping_from_pivot = mapping_from_pivot
        proto_space_for_mapping :Dict[str,StatSpace] = {}
        for sys_id, systems in members.items():
            proto_space_for_mapping[sys_id] = systems.state_space
        mapping = SymmetricStateSpaceMapping(pivot_element_name, mapping_from_pivot,proto_space_for_mapping)
        self.pivot_system = members[pivot_element_name]
        super().__init__(members, mapping,initial_state = self.pivot_system.initial_state,consistency_check = False)


    def multi_step_field_generator(self, steps: integer, curr_state: None, **params) -> Field:
        """Gnertates the a multi step field and and returns the multi-step field"""
        if curr_state is None:
            curr_state = self.initial_state
        return self.pivot_system.multi_step_field_generator(steps, curr_state)

    def save_multi_step_field(
            self, steps: integer, **params
    ):
        if self.repition:
            self.field = self.pivot_system.multi_step_field_generator(steps, **params)
            self.repition = False

    def evolve(self, steps: integer, **params):
        """Evolves the system by building l step field and applying the operator"""
        l_step_field = self.pivot_system.multi_step_field_generator(steps, **params)
        self.operator.apply(l_step_field, self.transition_list)
        self.initial_state = (self.transition_list[-1])
        self.field.set_unit_field_at_state(self.initial_state)
        self.repition = True

    def evolve_from_field(self):
        self.operator.apply(self.field, self.transition_list)
        self.initial_state = (self.transition_list[-1])
        self.field.set_field(self.transition_list[-1], self.field.get_unit_field())
        self.repition = True







