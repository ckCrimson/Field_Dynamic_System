from abc import ABC, abstractmethod

from fds.core.fds_field.single_field_value import SingleFieldValue


class Transform(ABC):
    """
    Abstract transformation holding input and output SingleFieldValue instances.
    """
    def __init__(self,**params):
        pass

    @abstractmethod
    def apply(self,input_field: SingleFieldValue) -> SingleFieldValue:
        """Perform the transformation from input_field to output_field."""
        pass


class NormTransform(Transform):
    """
    Compute the norm of a SingleFieldValue, storing result in output_field.
    """
    def __init__(self):
        super().__init__()

    def apply(self, inputFieldValue : SingleFieldValue) -> SingleFieldValue:
        # Compute norm of input tensor
        pass

