import numpy as np
from matplotlib import pyplot as plt

from Research_paper_systems_implementation.dice_system_implementation.field.dice_field_value import DiceRealFieldValue
from Research_paper_systems_implementation.dice_system_implementation.field.dice_gaussian_field_function import \
    GaussianFieldFunction
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_single_field_value import \
    DiceRealSingleFieldValue
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds import Field


class DiceRealField(Field):

    def __init__(self, state_space: DiceStateSpace,unit_field: DiceRealFieldValue=None , field_function: GaussianFieldFunction=None,
                 mean: float = None, variance : float = None,set_field_funtion:bool=False):
        unit_field= DiceRealFieldValue(DiceRealSingleFieldValue(1))
        if mean is None:
            mean = 3.5
        if variance is None:
            variance = 0.5
        field_function =GaussianFieldFunction(mean=mean, variance=variance)
        super().__init__(state_space, unit_field,field_function)

    def plot_field(self,title="",**kwargs) -> None:
        lower =self.state_space.lower_limit
        upper = self.state_space.upper_limit

        x_axis = np.linspace(lower,upper,100)
        y_axis = np.array([self.get_field(DiceState(x)).data.value for x in x_axis])
        plt.axvline((lower+upper)/2, color='r', linestyle='--',label=f" $ \mu $ = {(lower+upper)/2}")
        leg=plt.legend()
        plt.plot(x_axis,y_axis);plt.title(title);plt.show()

