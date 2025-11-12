from abc import ABC
from typing import Generic, TypeVar



from fds.core.fds_field.field_value import FieldValue
from fds.core.fds_state.state import State


class FunctionDefinition(ABC):

    def __init__(self, **params):
        self.param=params

    def function(self,state: State) -> FieldValue:
        pass


S = TypeVar('S', bound=State)
class FieldFunction(ABC, Generic[S]):
    """
    Interface defining dynamic or static mappings between states and fds_field values.
    Methods:
    - get_field(): returns the Field over the associated StateSpace.
    - get_field_at(state): returns the FieldValue at that state.
    - get_field_for_space(space): returns a Field defined over the given StateSpace.
    - function(state): core mapping for the fds_field value (can implement recurrence).
    """
    def __init__(self, func: FunctionDefinition ,**param):
        self.param = param
        self.func = func

    def get_field_at(self, state: S) -> FieldValue:
        """Return the FieldValue corresponding to the given state."""
        return self.func.function(state)

