import random

import numpy as np

from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_field import DiceRealField
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from fds.dynamics.fds_operator import Operator


class DiceOperator(Operator):
    def __init__(self):
        super().__init__()

    def get_next_state(self, field: DiceRealField) -> DiceState:
        lower = field.state_space.lower_limit
        upper = field.state_space.upper_limit
        x_axis = np.linspace(lower, upper, 1000)
        weight = np.array([field.get_field(DiceState(x)).data.value for x in x_axis])
        return DiceState(self._get_next_random_state(x_axis, weight))

    def _get_next_random_state(self, rv ,weights) -> int:
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Sample based on probabilities
        sampled_index = random.choices(range(len(rv)), weights=probabilities)[0]

        return rv[sampled_index]


