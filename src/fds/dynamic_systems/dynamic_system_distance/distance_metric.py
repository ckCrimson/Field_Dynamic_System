from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fds import State, Field

Sout = TypeVar('Sout', bound=State)

class DistanceMetric(ABC, Generic[Sout]):
    """
    Abstract metric defining distance between mapped and original fields.
    """
    @abstractmethod
    def metric(
        self,
        mapped_field: Field[Sout],
        field: Field[Sout]
    ) -> float:
        """
        Compute distance between a mapped field and the target field.
        """
        pass