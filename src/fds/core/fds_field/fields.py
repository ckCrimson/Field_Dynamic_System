
from typing import Generic, TypeVar, Optional

from fds.core.fds_state.state import State
from fds.core.fds_state.state_space import StatSpace

from fds.core.fds_field.field_function import FieldFunction
from fds.core.fds_field.field_value import FieldValue

S = TypeVar('S', bound=State)

class Field(Generic[S]):
    """
    Represents a fds_field over a state space, mapping states to FieldValue objects.
    """
    def __init__(self,
                 state_space: StatSpace[S],
                 unit_field: FieldValue,
                 field_function: Optional[FieldFunction[S]] = None):
        self.state_space = state_space
        self._unit = unit_field.get_unitary_field()
        self.field_function = field_function
        self._values: dict[S, FieldValue] = {}
        self.set_empty_field()

    def get_field(self, state: S) :
        """Return the FieldValue at the given state."""
        if state in self._values:
            return self._values[state]
        if self.field_function is not None:
            return self.field_function.get_field_at(state)

    def get_unit_field(self) -> FieldValue:
        return self._unit

    def get_zero_field(self) -> FieldValue:
        return self._unit.get_zero_field()

    def set_field(self, state: S, value: FieldValue) -> None:
        """Set the FieldValue at the given state."""
        self._values[state] = value

    def set_empty_field(self) -> None:
        self._values.clear()
        for s in self.state_space.get_all_states():
            self._values[s] = self._unit
        # Only store non-unit entries if needed; unit assumed default

    def set_zero_field(self):
        for s in self.state_space.get_all_states():
            self.set_field(s, self.get_zero_field())

    def add_field(self, other: 'Field[S]') -> None:
        """
        Add another fds_field into this one, composing values over common states.
        """
        # Ensure same state space
        if other.state_space is not self.state_space:
            raise ValueError("Cannot add fds_field with different state spaces")
        for state in self.state_space.get_all_states():
            fv1 = self.get_field(state)
            fv2 = other.get_field(state)
            self._values[state] = fv1.internal_composition(fv2.data)
    @property
    def unit(self):
        return self._unit

    def set_constant(self, f: FieldValue):
        for s in self.state_space.get_all_states():
            self.set_field(s,f)

    @classmethod
    def build_empty_from_state_space(self, cls ,state_space:StatSpace, unit_field:FieldValue=None) -> 'Field[S]':
        """Returns the empty Field corresponding to the given state space"""
        if None is unit_field:
            unit_field = cls.get_unit_field()
        return cls(state_space,unit_field)
    @classmethod
    def build_value_from_state_space(self, cls ,state_space:StatSpace, field_value:FieldValue) -> 'Field[S]':
        """Returns the empty Field corresponding to the given state space"""
        returnField =  cls(state_space,field_value.get_unitary_field())
        returnField.set_constant(field_value)
        return returnField

    def plot_field(self):
        """A method which can be implemented to plot the field if possible"""