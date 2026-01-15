from dataclasses import dataclass
from typing import Tuple, Any
from .interfaces import State


@dataclass(frozen=True)
class VectorState(State):
    values: tuple  # Type hint says tuple

    def __post_init__(self):
        # Force tuple conversion if user passes a list or array
        # This makes the object safely hashable.
        if not isinstance(self.values, tuple):
            # Bypass frozen=True using object.__setattr__
            object.__setattr__(self, 'values', tuple(self.values))

    def __repr__(self):
        return f"VectorState({self.values})"

    # --- ADD THESE ---
    def __eq__(self, other):
        if not isinstance(other, VectorState):
            return False
        return self.values == other.values

    def __hash__(self):
        return hash(self.values)

    def __add__(self, other):
        # Enable state + delta
        if isinstance(other, VectorState):
            new_vals = tuple(a + b for a, b in zip(self.values, other.values))
            return VectorState(new_vals)
        return NotImplemented


@dataclass(frozen=True)
class AbstractState(State):
    """
    Generic container for discrete user objects.
    """
    name: str
    properties: dict[str, Any]  # e.g. {'flammable': True}

    def __hash__(self):
        return hash(self.name)