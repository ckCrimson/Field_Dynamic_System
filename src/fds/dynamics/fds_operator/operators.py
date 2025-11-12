from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fds import Field, State

S = TypeVar('S', bound=State)
class Operator(ABC, Generic[S]):
    """
    Defines the system's state-transition operator.
    """
    def __init__(self,**params):
        self.params = params
    @abstractmethod
    def get_next_state(self, field: Field[S]) -> S:
        """Return the next observed state after l steps based on the field."""
        pass

    def apply(self, field: Field[S], history: list[State]) -> None:
        """Advance the system by one step: update state and record in history."""
        next_state = self.get_next_state(field)
        field.set_zero_field()
        field.set_field(next_state,field.get_unit_field())
        history.append(next_state)