from abc import abstractmethod
from typing import Generic, TypeVar, Optional

from numpy import integer

from fds import State, Field, StatSpace
from fds.core.fds_state import Reaching, Reachable
from fds.dynamic_systems.field_static_dynamic_system import FieldStaticDynamicSystem
from fds.dynamics.fds_operator.operators import Operator
from fds.dynamics.multi_step.multi_step_field import MultiStepField
from fds.dynamics.multi_step.multi_step_reaching import MultiStepReaching

S = TypeVar("S", bound=State)

class FieldDynamicSystem(FieldStaticDynamicSystem[S], Generic[S]):
    """
    Encapsulates a full Field Dynamic System with its components.

    Attributes:
      state_space: the state space
      field: the primary field over the state space
      multi_step_field: multi-step field generator
      reachable: Reachable instance
      reaching: Reaching instance
      operator: Operator defining system transitions
      transition_list: TransitionList storing history
      system_global_field: secondary global field inherited each iteration
    """
    def __init__(self, initial_state : State , field: Field[S], multi_step_field: MultiStepField[S], reachable: Reachable[S],
                 operator: Operator[S], transition_list: list[S]=None,state_space: StatSpace[S] = None,
                 reaching: Reaching[S]=None,
                 system_global_field: Optional[Field[S]] = None, own_produced_field: Optional[Field[S]] = None,
                 build_from_reachable = False,within_state_space = False, multi_step_reaching: MultiStepReaching=None) -> None:
        super().__init__(initial_state,state_space, reachable, field,reaching,multi_step_reaching,system_global_field,own_produced_field)
        self.multi_step_field = multi_step_field
       # self.pdf_field = pdf_field
        self.operator = operator
        if transition_list is None:
            transition_list: list[S]=[]
        self.transition_list = transition_list
        self.transition_list.append(initial_state)
        # If no initial_system provided, use self as initial,
        # Global field defaults to the primary field if not provided
        self.system_global_field = system_global_field or field
        self.multiStepReaching= MultiStepReaching(self.reachable)
        self.repition = True
        if state_space is None:
            if build_from_reachable:
                self.state_space = self.reachable.get_reachable(self.initial_state,within_state_space)
                self.field = self.field.build_value_from_state_space(type(self.field),self.state_space,self.field.get_zero_field())
                self.field.set_field(self.initial_state,self.field.get_unit_field())

    def back_to_initial(self, initial_system: Optional['FieldDynamicSystem[S]'] = None,) -> None:
        """Reset all components back to the initial system state."""
        init = initial_system
        self.state_space = init.state_space
        self.field = init.field
        self.multi_step_field = init.multi_step_field
       # self.pdf_field = init.pdf_field
        self.reachable = init.reachable
        self.reaching = init.reaching
        self.operator = init.operator
        self.transition_list = init.transition_list
        self.system_global_field = init.system_global_field

    @abstractmethod
    def multi_step_field_generator(self, steps:integer, curr_state: State=None ,**params) -> Field:
        """Gnertates the a multi step field and and returns the multi-step field"""

    def save_multi_step_field(
        self, steps:integer, **params
    ):
        if self.repition:
            self.field=self.multi_step_field_generator(steps, **params)
            self.repition  = False

    def evolve(self,steps: integer, **params):
        """Evolves the system by building l step field and applying the operator"""
        l_step_field = self.multi_step_field_generator(steps, **params)
        self.operator.apply(l_step_field,self.transition_list)
        self.initial_state =   (self.transition_list[-1])
        self.field.set_unit_field_at_state(self.initial_state)
        self.repition =  True

    def evolve_from_field(self):
        self.operator.apply(self.field,self.transition_list)
        self.initial_state =   (self.transition_list[-1])
        self.field.set_field(self.transition_list[-1],self.field.get_unit_field())
        self.repition = True
