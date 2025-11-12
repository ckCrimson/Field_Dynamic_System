import copy
from typing import TypeVar, Generic

from fds import State, Field
from fds.core.fds_field.fields_mapping.fields_mapping import FieldsMapping
from fds.dynamic_systems.dynamic_system_distance.distance_metric import DistanceMetric

Sin = TypeVar('Sin', bound=State)
Sout = TypeVar('Sout', bound=State)
class FieldsDistance(Generic[Sin, Sout]):
    """
    Computes distance between dynamic systems via field mapping and metric.
    """
    def __init__(
        self,
        fm: FieldsMapping[Sin, Sout],
        dm: DistanceMetric[Sout]
    ):
        self.fm = fm
        self.dm = dm

    def get_field_distance(
        self,
        field_in: Field[Sin],
        field_out: Field[Sout],
    ) -> float:
        """
        Map `field_in` to Field[Sout], fiel_out for the reference of output field then compute distance to `field`.
        Only possible if state-space and field-value types match.
        """
        # Obtain the mapped field
        field_out_type=copy.deepcopy(field_out)
        field_mapped: Field[Sout] = self.fm.get_fields_mapping(field_in,field_out_type)
        # Ensure the state spaces match
        if field_mapped.state_space != field_out.state_space:
            raise ValueError("Cannot compute distance: mismatched state spaces")
        # Ensure the field value types match (same unit field type)
        if not isinstance(field_mapped.get_unit_field(), field_out.get_unit_field().__class__):
            raise ValueError("Cannot compute distance: field value types must match exactly")
        # Compute and return the distance
        return self.dm.metric(field_mapped, field_out)


