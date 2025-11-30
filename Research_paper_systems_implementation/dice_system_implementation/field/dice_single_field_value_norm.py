from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_single_field_value import \
    DiceRealSingleFieldValue
from fds.core.fds_field.single_field_value import SingleFieldValue
from fds.core.fds_field.single_field_value_transform import Transform, NormTransform


class DiceFieldValueNorm(NormTransform):
    def __init__(self):
        super().__init__()

    def apply(self,input_field: DiceRealSingleFieldValue) -> DiceRealSingleFieldValue:
        return DiceRealSingleFieldValue(abs(input_field.value))

