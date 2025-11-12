from typing import TypeVar, Generic

from fds import State, Field
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
        ssm: StateSpaceMapping[Sin, Sout],
        d: Distribution[Sin, Sout], fields_transform: Transform
    ):
        self.ssm = ssm
        self.d = d
        self.field_transform = fields_transform

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
        pass
