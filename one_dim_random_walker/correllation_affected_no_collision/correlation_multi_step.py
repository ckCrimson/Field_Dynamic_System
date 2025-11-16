from typing import Dict, cast

from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace
from fds.dynamics.multi_step import MultiStepField
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealFieldMultiplication, \
    RealFieldAddition, RealFieldIdentityTransform, RealField
from one_dim_random_walker.correllation_affected_no_collision.correlation_reachable import \
    RandomWalkerCorrelatedReachable
from one_dim_random_walker.correllation_affected_no_collision.correlation_single_step import \
    RandomWalkerCorrelatedSingleStep
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem



class RandomWalkerCorrelatedMultiStepField(MultiStepField):
    def __init__(self, reachable : RandomWalkerCorrelatedReachable =None, single_step: RandomWalkerCorrelatedSingleStep=None,
                intrinsic_composer : RealFieldMultiplication =None, extrinsic_composer : RealFieldAddition = None,
                 global_field_transform: RealFieldIdentityTransform=None, single_step_transform: RealFieldIdentityTransform=None,
                 multi_step_transform: RealFieldIdentityTransform=None, dict_of_systems :Dict[str,OneDimWalkerFieldDynamicSystem]=None):
        if reachable is None:
            reachable = RandomWalkerCorrelatedReachable(dict_of_systems)
        if single_step is None:
            single_step = RandomWalkerCorrelatedSingleStep(dict_of_systems)
        if intrinsic_composer is None:
            intrinsic_composer = RealFieldMultiplication()
        if extrinsic_composer is None:
            extrinsic_composer = RealFieldAddition()
        if global_field_transform is None:
            global_field_transform = RealFieldIdentityTransform()
        if single_step_transform is None:
            single_step_transform = RealFieldIdentityTransform()
        if multi_step_transform is None:
            multi_step_transform = RealFieldIdentityTransform()
        super().__init__(reachable,single_step, intrinsic_composer, extrinsic_composer, global_field_transform, single_step_transform, multi_step_transform)

    def generate_multi_step_field(
            self,
            space: CorrelatedStateSpace,  # ambient/type anchor
            start: CorrelatedState,
            L: int,
            single_transformed_field_proto: RealField = None,
            single_step_transformed_field_proto:RealField= None,
            prev_field_input: RealField = None,  # may be mutated (as you allowed)
            global_field: RealField = None
    ) -> RealField:
      return cast(RealField, super().generate_multi_step_field(space, start, L, single_transformed_field_proto, single_step_transformed_field_proto,prev_field_input,global_field))