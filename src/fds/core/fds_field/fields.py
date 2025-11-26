
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
        self.constant_field=self._unit
        self._zero_value = self._unit.get_zero_field()
        self.constant_indicator=False

    def get_field(self, state: S) :
        """Return the FieldValue at the given state."""
        if state in self._values:
            return self._values[state]
        if self.field_function is not None:
            if self.constant_indicator:
                return self.constant_field
            return self.field_function.get_field_at(state)

    def get_unit_field(self) -> FieldValue:
        return self._unit

    def get_zero_field(self) -> FieldValue:
        return self._unit.get_zero_field()

    def set_field(self, state: S, value: FieldValue=None, field_function:FieldFunction[Optional]=None) -> None:
        """Set the FieldValue at the given state."""
        if value is not None:
            self._values[state] = value
        if field_function is not None:
            self.constant_indicator=False
            self.field_function = field_function


    def set_empty_field(self) -> None:
        if self.field_function is not None:
            self.constant_indicator=True
            self.constant_field=self._unit
        else:
            ids = self.state_space.ids_view()
            for id in ids:
                self.set_field(self.state_space.get_state_by_id(int(id)), self._unit)
        # Only store non-unit entries if needed; unit assumed default

    def set_zero_field(self):
        if self.field_function is not None:
            self.constant_indicator=True
            self.constant_field=self._zero_value
        else:
            ids = self.state_space.ids_view()
            for id in ids:
                self.set_field(self.state_space.get_state_by_id(int(id)), self._zero_value)

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
        if self.field_function is not None:
            self.constant_indicator=True
            self.constant_field=f
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

    def set_unit_field_at_state(self,inp_state:State):
        if self.field_function is not None:
            self.constant_field = self._unit
        self.set_zero_field()
        self.set_field(inp_state,self.get_unit_field())

    def set_constant_field_at_all_except_input(self, inp_state:State, inp_constant_field :FieldValue, inp_special_field :FieldValue):
        if self.field_function is not None:
            self.constant_indicator=True
            self.constant_field=inp_constant_field
            self.set_field(inp_state,inp_special_field)
        else:
            ids = self.state_space.ids_view()
            no_check =False
            for id in ids:
                state = self.state_space.get_state_by_id(int(id))
                if not no_check and state == inp_state:
                    self.set_field(state,inp_special_field)
                    no_check = True
                    continue
                self.set_field(state,inp_constant_field)

    def set_zero_everywhere_except_unit_at_inp_state(self, inp_state:State):
        un =self._unit
        zer = self._zero_value
        self.set_constant_field_at_all_except_input(inp_state,zer, un)