from fds.core.fds_field.single_field_value import SingleFieldValue


class DiceRealSingleFieldValue(SingleFieldValue):

    value: float

    def __str__(self):
        return f'{self.value}'
