from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_single_field_value import \
    DiceRealSingleFieldValue
from Research_paper_systems_implementation.dice_system_implementation.field.dice_single_field_value_addition import \
    DiceFieldValueAddition
from Research_paper_systems_implementation.dice_system_implementation.field.dice_single_field_value_norm import \
    DiceFieldValueNorm
from fds import FieldValue


class DiceRealFieldValue(FieldValue):
    data: DiceRealSingleFieldValue


    def get_unitary_field(self) -> 'DiceRealFieldValue':
        return DiceRealFieldValue(DiceRealSingleFieldValue(1))

    def get_zero_field(self) -> "DiceRealFieldValue":
        return DiceRealFieldValue(DiceRealSingleFieldValue(0))

