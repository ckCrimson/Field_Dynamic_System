from abc import ABC, abstractmethod

from fds.core.fds_field.single_field_value import SingleFieldValue


class Composition(ABC):
    """
    Abstract composition operator combining two FieldValue inputs into one output.
    """

    def __init__(self, **params):
        pass

    @abstractmethod
    def compose(self, input1: SingleFieldValue, input2: SingleFieldValue) -> SingleFieldValue:
        """
        Combine input1 and input2 into a new FieldValue.
        """
        pass


class AdditionComposition(Composition):
    """
    Composition that adds two real-valued FieldValue tensors elementwise.
    """

    def __init__(self):
        super().__init__()

    def compose(self, input1: SingleFieldValue, input2: SingleFieldValue) -> SingleFieldValue:
        # Elementwise addition of underlying tensors
        pass
