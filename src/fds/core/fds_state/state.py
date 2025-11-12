from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import  TypeVar, Generic
from typing import Any


@dataclass(slots=True, frozen=True, eq=True)
class State:
    """
    Minimal immutable state container.
    `value` must be hashable (e.g., int, str, tuple[int,...]).
    Designed for fast dict/set use and zero-cost copying.
    """
    state: Any  # prefer: int | str | tuple[int, ...] for speed & hashability

    def __post_init__(self):
        # Fail fast if value is not hashable (prevents subtle runtime issues)
        try:
            hash(self.state)
        except TypeError as e:
            raise TypeError("State.value must be hashable") from e

    # Value-based hash/equality come from dataclass(frozen=True, eq=True)
    # For builtins like tuple/int/str, CPython already caches their hashes.

    # Zero-cost copy APIs (important if user code accidentally copies states)
    def __copy__(self) -> "State":       # shallow copy
        return self
    def __deepcopy__(self, memo) -> "State":  # deep copy
        return self

    # Optional niceties — keep cheap
    def __len__(self) -> int:
        # Treat scalars as dim=1; tuples expose their length
        v = self.state
        if isinstance(v, tuple):
            return len(v)
        return 1

    def __repr__(self) -> str:
        # Short and log-friendly
        return f"State({self.state!r})"

    # Convenience for integer-like states (optional)
    def __int__(self) -> int:
        v = self.state
        if isinstance(v, int):
            return v
        if isinstance(v, tuple) and len(v) == 1 and isinstance(v[0], int):
            return v[0]
        raise TypeError("State is not integer-like")

S = TypeVar('S', bound=State)

class StateOperations(ABC, Generic[S]):
    """
    Interface for generic operations on states.
    Implementations define concrete operations.*"""
    @abstractmethod
    def apply(self, a: S, b: S):
        """Run the operation on two states."""
        raise NotImplementedError

