from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_single_field_value import \
    DiceRealSingleFieldValue
from fds.core.fds_field.single_field_value_composition import AdditionComposition


class DiceFieldValueAddition(AdditionComposition):
    def __init__(self):
        super().__init__()

    def compose(self, input1: DiceRealSingleFieldValue, input2: DiceRealSingleFieldValue) -> DiceRealSingleFieldValue:
        return DiceRealSingleFieldValue(input1.value + input2.value)
