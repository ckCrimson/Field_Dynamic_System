from typing import Optional

from Research_paper_systems_implementation.dice_system_implementation.field.dice_gaussian_field_function import \
    GaussianFieldFunction
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_field import DiceRealField
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reachable import DiceReachable
from fds import Field
from fds.dynamics.single_step import SingleStepField
from fds.dynamics.single_step.single_step_field import S


class DiceSingleStepFieldGenerator(SingleStepField):
    def __init__(self, reachable: DiceReachable=None,alpha: int=25, n_0: int =100, dice_number: int=6,type:str = "C"):
        if reachable is None:
            reachable = DiceReachable(alpha,n_0,dice_number)
        else :
            reachable = reachable
        self.type= type
        self.dice_number = dice_number
        self.alpha = alpha
        self.n_0 = n_0
        super().__init__(reachable=reachable)

    def build_single_step_field(
        self,
        current_state: S,
        system_field: Field[S],
        global_field: "Field[S]"=None,
        q_field: Optional["Field[S]"] = None,   # <-- NEW: optional shaping field
    ) -> DiceRealField:
        reachable_space = self.reachable.get_reachable(current_state)
        lower = reachable_space.lower_limit
        upper = reachable_space.upper_limit
        mean = (lower+upper)/2
        variance = 0.5
        if self.type == "C":
            variance = (self.dice_number-1)*(self.dice_number+self.alpha)*self.alpha/12
        else:
            variance = ((self.dice_number**2)-1)*self.alpha/12
        field_function = GaussianFieldFunction(mean=mean, variance=variance)
        dice_real_field = DiceRealField(reachable_space)
        dice_real_field.set_field_function(field_function)
        return dice_real_field