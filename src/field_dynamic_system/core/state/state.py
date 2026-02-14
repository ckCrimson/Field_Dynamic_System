from dataclasses import dataclass, field
from typing import Any, Dict
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
    name: Any
    properties: Dict[Any, Any] = field(default_factory=dict) # e.g. {'flammable': True}

    @property
    def value(self) -> Any:
        """
        Alias: Exposes 'name' as 'value'.
        Fixes legacy naming inconsistency without breaking changes.
        """
        return self.name

    @property
    def values(self) -> Any:
        """
        Alias: Exposes 'name' as 'value'.
        Fixes legacy naming inconsistency without breaking changes.
        """
        return self.name

    def __hash__(self):
        return hash(self.name)