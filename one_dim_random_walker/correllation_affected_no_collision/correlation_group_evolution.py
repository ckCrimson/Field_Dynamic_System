from typing import Dict

from fds import Field
from fds.affecting.affecting_systems_framework.affecting_groups import InterAffectingGroups
from fds.affecting.affecting_systems_framework.grouped_evolution import AffectingGroupsEvolution
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField
from one_dim_random_walker.correllation_affected_no_collision.correlation_affected_fds import \
    OneDimRWCorrelatedAffectingFDS
from one_dim_random_walker.correllation_affected_no_collision.correlation_grouping import CorrelationAffectingGroups
from one_dim_random_walker.correllation_affected_no_collision.correlation_mapping import CorrerlatedAffectingRWMapping
from one_dim_random_walker.correllation_affected_no_collision.is_correlated import IsRWCorrelated
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem

'''def __init__(
        self,
        systems: Dict[str, FieldDynamicSystem],
        group_generator: InterAffectingGroups,
        mapping: AffectedSystemsMapping =None,
        state_space: StatSpace = None,
        field: Field  = None,
        reachable: AffectedReachable[S] = None,
        transition_list: Optional[List[S]] = None,
        reaching: Optional[AffectedReaching] = None,
        multi_step_field: Optional[MultiStepField] = None,
        operator: Optional[AffectedSystemsOperator] = None,
        system_global_field: Optional[Field] = None,
        sync_members_from_s0: bool = False,  # if True and s0 provided, push s0 → members
        all_independent_systems: bool = False,'''

class OneDimRWCorrelatedGroupEvolution(AffectingGroupsEvolution):
    def __init__(self, systems :Dict[str,OneDimWalkerFieldDynamicSystem],
                 group_generator: CorrelationAffectingGroups=None,
                 mapping: CorrerlatedAffectingRWMapping = None
    ):
        if group_generator is None:
            group_generator = CorrelationAffectingGroups(systems,IsRWCorrelated())
        if mapping is None:
            mapping =   CorrerlatedAffectingRWMapping()
        super().__init__(systems,group_generator,mapping)

    def form_affected_system_from_dict( self,
        group: Dict[str, OneDimWalkerFieldDynamicSystem]
    ) -> OneDimRWCorrelatedAffectingFDS:
        return OneDimRWCorrelatedAffectingFDS(group)