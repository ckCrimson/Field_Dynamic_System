import numpy as np

from Research_paper_systems_implementation.dice_system_implementation.field.dice_field_value import DiceRealFieldValue
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_single_field_value import \
    DiceRealSingleFieldValue
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds.core.fds_field.field_function import FieldFunction


class GaussianFieldFunction(FieldFunction):

    def __init__(self,mean:float,variance:float):
        self.mean = mean
        self.variance = variance
        super().__init__()

    def get_field(self,state: DiceState) -> DiceRealFieldValue:
        if np.any(np.array(self.variance) <= 0):
            raise ValueError("Variance must be strictly positive (> 0).")

            # The standard deviation is the square root of the variance
        std_dev = np.sqrt(self.variance)

        # 2. Left (Normalization) Term: 1 / sqrt(2 * pi * variance)
        # Using numpy's constants is preferred
        normalization_factor = 1 / (std_dev * np.sqrt(2 * np.pi))

        # 3. Exponent Term: exp(-((c - mean)^2) / (2 * variance))
        exponent = -0.5 * (((state.state - self.mean) / std_dev) ** 2)

        # 4. Final Calculation: Normalization * Exponent Term
        pdf_value = normalization_factor * np.exp(exponent)

        return DiceRealFieldValue( DiceRealSingleFieldValue( pdf_value) )

