from abc import abstractmethod, ABC

from fds.core.fds_field.single_field_value import SingleFieldValue
from fds.core.fds_field.single_field_value_composition import Composition


class KernelFieldComposition(Composition,ABC):  # extends your Composition
    def __init__(self, **param):     # <-- fix: __init__, not __int__
        super().__init__(**param)

    @abstractmethod
    def compose(self, input1: "SingleFieldValue", input2: "SingleFieldValue") -> "SingleFieldValue":
        pass
