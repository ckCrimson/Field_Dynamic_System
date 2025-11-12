from typing import TypeVar, Generic

from fds import State
from fds.dynamic_systems.dynamic_system_distance.fields_distance import FieldsDistance
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem

S = TypeVar('S', bound=State)

class FDSDistance(Generic[S]):
    """
    Computes distance between two field dynamic systems using a FieldsDistance.
    """
    def __init__(self, fd: FieldsDistance[S, S]):
        self.fd = fd

    def get_system_distance(
        self,
        fds1: FieldDynamicSystem[S],
        fds2: FieldDynamicSystem[S]
    ) -> float:
        """
        Compute distance between two systems by comparing their fields.
        """
        return self.fd.get_field_distance(fds1.field, fds2.field)
