from typing import Optional, Dict

from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace
from fds.dynamics.single_step import SingleStepField, Kernel
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldValue, \
    RealSingleFieldValue
from one_dim_random_walker.correllation_affected_no_collision.correlation_reachable import \
    RandomWalkerCorrelatedReachable
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem


class RandomWalkerCorrelatedSingleStep(SingleStepField):
    def __init__(self,systems: Dict[str,OneDimWalkerFieldDynamicSystem], reachable :RandomWalkerCorrelatedReachable=None):
        self.systems =systems
        if reachable is None:
            reachable = RandomWalkerCorrelatedReachable(self.systems)
        super().__init__(reachable = reachable )

    def build_single_step_field(self,
                                current_state: CorrelatedState,
                                system_field: RealField,
                                global_field: RealField = None,
                                q_field: Optional[RealField] = None,  # <-- NEW: optional shaping field
                                ) -> RealField:

        reachable_space = self.reachable.get_reachable(current_state)
        return_field = RealField.build_value_from_state_space(  RealField,
                                                                reachable_space,
                                                                RealFieldValue(RealSingleFieldValue(1))
                                                                )
        ids = reachable_space.ids_view()
        for sid in ids:
            state  = reachable_space.get_state_by_id(int(sid))
            field_at_state = RealFieldValue(RealSingleFieldValue(0))
            if isinstance(reachable_space,CorrelatedStateSpace):
                field_at_state = field_at_state+reachable_space.dimension()
            return_field.set_field(state,field_at_state)
        return return_field