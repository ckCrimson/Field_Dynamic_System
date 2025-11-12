from typing import TypeVar, Optional

from fds.core.fds_field.single_field_value import SingleFieldValue
from fds.dynamics.single_step import Kernel, SingleStepField
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealFieldValue, \
    RealFieldValueMultiplication, RealFieldValueAddition, RealField
from one_dim_random_walker.core.states.reachable import OneDimensionReachable
from one_dim_random_walker.core.states.state import IntegerState

I = TypeVar("I", bound=IntegerState)

# -----------------Kernel ------------------#


class OneDimUniformKernel(Kernel):
    def __init__(self):
        super().__init__()

    def get_kernel_value(self, state: I, current_state: I) -> "RealFieldValue":
        return RealFieldValue(SingleFieldValue(1))

class SkewedKernel(Kernel):
    def __init__(self, bias_value=0):
        params={}
        params['bias'] = bias_value
        self.bias=bias_value
        super().__init__()

    def get_kernel_value(self, state: I, current_state: I) -> "RealFieldValue":
        if state.state-current_state.state < 0:
            return RealFieldValue(SingleFieldValue(self.bias))
        return RealFieldValue(SingleFieldValue(1))

# ----------- Single Step ---------------#



class OneDimSingleStep(SingleStepField):
    def __init__(self, kernel_type= "U", bias_value=1, reachable_distance =  1, kernel_comp_type = 'M' ):
        """kerrnel type : U -> uniform else Skewed, bias balue required in else, reachable_distance = 1, kernel_comp_type = M"""
        if "U"==kernel_type:
            kernel = OneDimUniformKernel()
        else:
            if None is bias_value:
                raise ValueError("bias value cannot be None")
            kernel = SkewedKernel(bias_value)
        reachable =   OneDimensionReachable(reachable_distance)
        if "M"==kernel_comp_type:
            kernel_composition = RealFieldValueMultiplication()
        else:
            kernel_composition = RealFieldValueAddition()
        super().__init__(kernel,reachable, kernel_composition)
        self.reachable_distance=reachable_distance

    def build_single_step_field(
        self,
        current_state: IntegerState,
        system_field: RealField,
        global_field:RealField= None,
        q_field: Optional[RealField] = None,   # <-- NEW: optional shaping field
    ) -> RealField:
       return RealField.from_length(self.reachable_distance,current_state)




