from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from fds import State

S = TypeVar('S', bound=State)
class StateOperations(ABC, Generic[S]):
    """
    Interface for generic operations on states.
    Implementations define concrete operations.*"""
    @abstractmethod
    def apply(self, a: S, b: S):
        """Run the operation on two states."""
        raise NotImplementedError