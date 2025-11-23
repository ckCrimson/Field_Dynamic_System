from typing import TypeVar, Generic

from fds import State, Field
from fds.core.fds_field import ComposeField
from fds.core.fds_field.fields_mapping.distributions import Distribution
from fds.core.fds_field.single_field_value_transform import Transform
from fds.core.fds_state import StateSpaceMapping

Sin = TypeVar('Sin', bound=State)
Sout = TypeVar('Sout', bound=State)

class FieldsMapping(Generic[Sin, Sout]):
    """
    Maps an entire Field from one state-space to another using a state-space mapping
    and a distribution function.
    """
    def __init__(
        self,
        fields_transform: Transform,
        field_composition: ComposeField ,
        state_space_mapping: StateSpaceMapping[Sin, Sout] = None,
        distribution: Distribution[Sin, Sout] = None,
    ):
        self.fields_transform = fields_transform
        self.field_composition = field_composition
        self.state_space_mapping = state_space_mapping
        self.distribution = distribution


    def get_fields_mapping(
        self,
        field_in: Field[Sin],
        field_out: Field[Sout]
    ) -> Field[Sout]:
        """
        Map a Field[Sin] to Field[Sout].
        """
        # Initialize empty output field over the union of all mapped spaces
        # First compute the overall output state-space
        state_space_in = field_in.state_space

        field_2_temp = field_out.build_value_from_state_space(field_out.__class__, field_out.state_space, field_out.get_zero_field())
        ids= state_space_in.ids_view()
        for sid in ids:
            state_in = state_space_in.get_state_by_id(int(sid))
            f_2_state_in = self.distribution.distributionFunction(field_in.get_field(state_in),field_2_temp)
            field_2_temp = self.field_composition.apply(field_2_temp,f_2_state_in)

        return field_2_temp
