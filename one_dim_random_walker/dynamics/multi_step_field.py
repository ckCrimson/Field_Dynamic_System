
from fds.dynamics.multi_step import MultiStepField
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealFieldAddition, \
    RealFieldMultiplication, RealFieldIdentityTransform
from one_dim_random_walker.core.states.reachable import OneDimensionReachable

from one_dim_random_walker.dynamics.single_step import OneDimSingleStep

class OneDimRandomWalkerMultiStep(MultiStepField):
    def __init__(self, distance: int = 1, kernel_type = 'U', bias_value=1,):
        reachable = OneDimensionReachable(distance)
        if kernel_type == 'U':
            single_step_field= OneDimSingleStep(reachable_distance=distance)
        else:
            single_step_field= OneDimSingleStep(reachable_distance = distance,kernel_type = "B", bias_value = bias_value)
        intrinsic_composer = RealFieldMultiplication()
        extrinsic_composer = RealFieldAddition()
        global_field_transform = RealFieldIdentityTransform()
        single_step_transform = RealFieldIdentityTransform()
        multi_step_transform = RealFieldIdentityTransform()
        super().__init__(
            reachable= reachable,
            single_step= single_step_field,
            intrinsic_composer= intrinsic_composer,  # Z^P  (combine parent value with step field)
            extrinsic_composer= extrinsic_composer,  # Z^t  (accumulate contributions from different parents)
            global_field_transform=global_field_transform,  # optional transform of global field each step
            single_step_transform =single_step_transform,  # transform for the single-step field
            multi_step_transform=multi_step_transform  # transform for the multi-stepped field
        )