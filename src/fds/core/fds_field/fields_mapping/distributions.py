from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fds import State, Field, StatSpace
from fds.core.fds_state import StateSpaceMapping

Sin = TypeVar('Sin', bound=State)
Sout = TypeVar('Sout', bound=State)

class Distribution(ABC, Generic[Sin, Sout]):
    """
    Abstract distribution defining how field values at an input state spread over
    the mapped output state-space.
    """
    def __init__(self,stateSpaceMapping: StateSpaceMapping[Sin, Sout],):
        self.stateSpaceMapping = stateSpaceMapping
    @abstractmethod
    def distributionFunction(
        self,
        field_in: Field[Sin],
        field_out: Field[Sout]
    ) -> Field[Sout]:
        """
        Given an input field and a single input state plus its mapped output space,
        return a Field over that output space representing the distribution.
        """
        pass