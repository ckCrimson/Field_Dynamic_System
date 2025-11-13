from fds.core.fds_state import Reaching
from fds.dynamic_systems import FieldDynamicSystem
from fds.dynamics.fds_operator import Operator
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField
from one_dim_random_walker.core.states.reachable import OneDimensionReachable
from one_dim_random_walker.core.states.state import IntegerState
from one_dim_random_walker.core.states.state_space import IntegerLine
from one_dim_random_walker.dynamics.multi_step_field import OneDimRandomWalkerMultiStep


'''def __init__(self, initial_state : State , field: Field[S], multi_step_field: MultiStepField[S], reachable: Reachable[S],
                 operator: Operator[S], transition_list: list[S],state_space: StatSpace[S] = None,
                 reaching: Reaching[S]=None,
                 system_global_field: Optional[Field[S]] = None, own_produced_field: Optional[Field[S]] = None,
                 build_from_reachable = False,within_state_space = False) -> None:'''

class  OneDimWalkerFieldDynamicSystem(FieldDynamicSystem):
    def __init__(self, initial_state:IntegerState, field: RealField, multi_step: OneDimRandomWalkerMultiStep,
                 reachable: OneDimensionReachable, operator: Operator,transition_list : list ,state_space:IntegerLine=None,
                 reaching: Reaching=None,system_global_field: RealField=None,own_produced_field=None):
        super().__init__(initial_state,field,multi_step,reachable, operator, transition_list, state_space, reaching, system_global_field, own_produced_field)