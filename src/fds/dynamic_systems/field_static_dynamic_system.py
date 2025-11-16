import copy
from typing import Generic, TypeVar, Optional

from fds import State, StatSpace, Field
from fds.core.fds_state import Reachable, Reaching
from fds.dynamic_systems.static_dynamic_systems import StaticDynamicSystem
from fds.dynamics.multi_step import MultiStepReaching

T = TypeVar("T", bound=State)

class FieldStaticDynamicSystem(StaticDynamicSystem[T], Generic[T]):
    """
    Static system that also carries a field over its state space.
    """
    def __init__(
        self,
        initial_state: State,
        state_space: StatSpace[T],
        reachable: Reachable[T],
        field: Field[T],
        reaching: Reaching[T]=None,
        multi_step_reaching: MultiStepReaching[T]=None,
        global_system_field: Optional[Field[T]] = None,
        own_produced_global_field: Optional[Field[T]] = None
    ):
        super().__init__(initial_state,state_space, reachable, reaching,multi_step_reaching=multi_step_reaching)
        self.field: Field[T] = field
        self.own_produced_global_field: Field[T] = own_produced_global_field
        if global_system_field is not None and global_system_field is None:
            self.own_produced_field = global_system_field
        if global_system_field is None:
            # assume `field._unit` holds the unit FieldValue
            self.global_field = copy.deepcopy(field)
            self.global_field.set_empty_field()
        else:
            self.global_field = global_system_field
        self.transition_list = []
