from Research_paper_systems_implementation.dice_system_implementation.dynamics.single_step.single_step_field_generator import \
    DiceSingleStepFieldGenerator
from Research_paper_systems_implementation.dice_system_implementation.field.dice_gaussian_field_function import \
    GaussianFieldFunction
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_field import DiceRealField
from Research_paper_systems_implementation.dice_system_implementation.state.dice_multi_step_reaching import \
    DiceMultiStepReaching
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reachable import DiceReachable
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds.dynamics.multi_step import MultiStepField


class DiceMultiStepFieldGenerator(MultiStepField):
    def __init__(self, reachable: DiceReachable=None, single_step: DiceSingleStepFieldGenerator=None, alpha: int=25, n_0: int = 100,
                 dice_number:int=6, type:str ='C'):
        self.alpha = alpha
        self.n_0 = n_0
        self.dice_number = dice_number
        if reachable is None:
            reachable = DiceReachable(alpha,n_0,dice_number)
        if single_step is None:
            single_step = DiceSingleStepFieldGenerator(reachable,alpha,n_0,dice_number,type)
        super().__init__(reachable,single_step)

    def generate_multi_step_field(
            self,
            space: DiceStateSpace,  # ambient/type anchor
            start: DiceState,
            L: int,
            single_transformed_field_proto: DiceRealField = None,
            single_step_transformed_field_proto: DiceRealField = None,
            prev_field_input: DiceRealField = None,  # may be mutated (as you allowed)
            global_field:DiceRealField= None
    ) -> DiceRealField:
        multi_step_generator = DiceMultiStepReaching(self.alpha,self.n_0,self.dice_number)
        reachable_space = multi_step_generator.get_multi_step_reaching(start,L)
        lower = reachable_space.lower_limit
        upper = reachable_space.upper_limit
        mean = (lower + upper) / 2
        variance = 0.5
        if self.single_step.type == "C":
            variance = (self.dice_number - 1) * (self.dice_number + self.alpha) * self.alpha / 12
        else:
            variance = ((self.dice_number ** 2) - 1) * self.alpha / 12
        field_function = GaussianFieldFunction(mean=mean, variance=variance)
        dice_real_field = DiceRealField(reachable_space)
        dice_real_field.set_field_function(field_function)
        return dice_real_field