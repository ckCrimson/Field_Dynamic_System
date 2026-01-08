from dataclasses import dataclass
from typing import Tuple, Any
from .interfaces import State


@dataclass(frozen=True)
class VectorState(State):
    """
    Concrete implementation for continuous vectors.
    """
    values: Tuple[float, ...]


@dataclass(frozen=True)
class AbstractState(State):
    """
    Generic container for discrete user objects.
    """
    name: str
    properties: dict[str, Any]  # e.g. {'flammable': True}

    def __hash__(self):
        return hash(self.name)