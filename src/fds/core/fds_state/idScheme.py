from __future__ import annotations
from typing import Protocol, Iterable, Sequence, Callable, Hashable, Generic, TypeVar
from dataclasses import dataclass
import numpy as np

S = TypeVar("S")

# ---------- Interface ----------

class IdScheme(Protocol, Generic[S]):
    """Invertible mapping between State objects and primitive integer IDs."""
    def universe_size(self) -> int: ...
    def to_id(self, state: S) -> int: ...
    def to_state(self, id_: int) -> S: ...
    def ids_view(self) -> np.ndarray: ...
    # Optional (fast bulk helpers; not used in hot loops)
    def to_ids(self, states: Iterable[S]) -> np.ndarray: ...
    def to_states(self, ids: Iterable[int]) -> list[S]: ...


# ---------- Default, general-purpose scheme ----------

@dataclass(slots=True)
class ExplicitIdScheme(Generic[S], IdScheme[S]):
    """
    Generic invertible mapping for ANY finite set of states.
    - States can be hashable, OR provide key_fn(state)->Hashable for lookup.
    - IDs are 0..N-1 in deterministic order of 'states' (or user-chosen order).
    """
    states: Sequence[S]
    key_fn: Callable[[S], Hashable] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.states, (list, tuple)):
            self.states = tuple(self.states)
        # Build forward/reverse maps
        self._id_to_state: list[S] = list(self.states)
        if self.key_fn is None:
            # Directly use the state as the dict key (requires hashable states)
            self._state_to_id: dict[Hashable, int] = {s: i for i, s in enumerate(self._id_to_state)}
        else:
            # Use a stable derived key (works for unhashable / complex states)
            self._state_to_id = {self.key_fn(s): i for i, s in enumerate(self._id_to_state)}
        # Contiguous ids view for fast iteration
        self._ids = np.arange(len(self._id_to_state), dtype=np.int32)

    # ---- IdScheme API ----
    def universe_size(self) -> int:
        return len(self._id_to_state)

    def to_id(self, state: S) -> int:
        return self._state_to_id[state if self.key_fn is None else self.key_fn(state)]

    def to_state(self, id_: int) -> S:
        if id_ < 0 or id_ >= len(self._id_to_state):
            raise IndexError("ID out of range")
        return self._id_to_state[id_]

    def ids_view(self) -> np.ndarray:
        return self._ids  # contiguous, read-only by convention

    # ---- Optional bulk helpers (convenience) ----
    def to_ids(self, states: Iterable[S]) -> np.ndarray:
        if self.key_fn is None:
            return np.fromiter((self._state_to_id[s] for s in states), dtype=np.int32)
        return np.fromiter((self._state_to_id[self.key_fn(s)] for s in states), dtype=np.int32)

    def to_states(self, ids: Iterable[int]) -> list[S]:
        out: list[S] = []
        for i in ids:
            if i < 0 or i >= len(self._id_to_state):
                raise IndexError("ID out of range")
            out.append(self._id_to_state[i])
        return out

class HashIntegerIdScheme(IdScheme):
    pass