from typing import Dict, List

from numpy import integer

from fds.affecting.affecting_systems_framework.affecting_fds import AffectingFDS
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace
from fds.core.fds_state import Reaching
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField
from one_dim_random_walker.correllation_affected_no_collision.correlation_grouped_operators import \
    CorrelationGroupOperators
from one_dim_random_walker.correllation_affected_no_collision.correlation_mapping import CorrerlatedAffectingRWMapping
from one_dim_random_walker.correllation_affected_no_collision.correlation_multi_step import \
    RandomWalkerCorrelatedMultiStepField
from one_dim_random_walker.correllation_affected_no_collision.correlation_reachable import \
    RandomWalkerCorrelatedReachable
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem

'''
def __init__(
        self,
        members: Dict[str, FieldDynamicSystem] = {},
        mapping: AffectedSystemsMapping = None,
        state_space: StatSpace = None,
        field: Field  = None,
        reachable: AffectedReachable[S] = None,
        transition_list: Optional[List[S]] = None,
        initial_state: Optional[S] = None,
        reaching: Optional[AffectedReaching] = None,
        multi_step_field: Optional[MultiStepField] = None,
        operator: Optional[AffectedSystemsOperator] = None,
        system_global_field: Optional[Field] = None,
        build_from_reachable:bool=False,
        within_state_space:bool=False,
        sync_members_from_s0: bool = False  # if True and s0 provided, push s0 → members
    ) -> None:'''
class OneDimRWCorrelatedAffectingFDS(AffectingFDS):
    def __init__(self, members :Dict[str,OneDimWalkerFieldDynamicSystem], mapping: CorrerlatedAffectingRWMapping = None,
                 state_space: CorrelatedStateSpace = None,
                 field: RealField  = None,reachable : RandomWalkerCorrelatedReachable =None,
                 transition_list:List = None ,initial_state: CorrelatedState = None,reaching: Reaching = None,
                 multi_step_field: RandomWalkerCorrelatedMultiStepField = None, operator: CorrelationGroupOperators=None,
                 ):
        if initial_state is None:
            initial_state = CorrelatedState.from_initial_states_of_systems(members)
        if mapping is None:
            mapping = CorrerlatedAffectingRWMapping()
        if state_space is None:
            state_space = CorrelatedStateSpace.build_from_dict_of_states(initial_state.get_all_components())
        if field is None:
           field =  RealField(state_space, initial_state)
           field.set_zero_everywhere_except_unit_at_inp_state(initial_state)
        if reachable is None:
            reachable = RandomWalkerCorrelatedReachable(members)
        if multi_step_field is None:
            multi_step_field = RandomWalkerCorrelatedMultiStepField(dict_of_systems=members)
        if operator is None:
            operator = CorrelationGroupOperators(members)
        super().__init__(members,mapping,state_space,field,reachable,None,initial_state,reaching,multi_step_field, operator)



    def multi_step_field_generator(self, steps: integer, curr_state: CorrelatedState= None, **params) -> RealField:
        if curr_state is None:
            curr_state=self.initial_state
        return self.multi_step_field.generate_multi_step_field(
            space = self.state_space,
            start= curr_state,
            L=int(steps),
            single_step_transformed_field_proto=RealField(self.state_space, curr_state),
            single_transformed_field_proto=RealField(self.state_space, curr_state),
            prev_field_input=self.field
        )