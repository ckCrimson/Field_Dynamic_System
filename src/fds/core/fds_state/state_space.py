from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, Any, List
from typing import Generic, TypeVar, Optional, Iterable

import numpy as np

from src.fds.core.fds_state.state import State


S = TypeVar('S', bound=State)

# -----------State Space ---------------#

class StatSpace(Generic[S],ABC):
    """
    Abstract state space with current state management and construction methods.
    The initial state is provided at instantiation.
    """
    def __init__(self, initial_state: S,
                 set_of_states: Iterable[State]=None,
    ):
        self._current_state: S = initial_state
        self.build_from_states(set_of_states,initial_state)


    def build_from_states(self, states: Iterable[S], current: Optional[S] = None) -> "StatSpace[S]":
        """Construct (or reconstruct) the universe deterministically from 'states'."""
        ...

    # ---- core fast paths ----
    @abstractmethod
    def size(self) -> int:
        ...


    def ids_view(self) -> np.ndarray:
        """Contiguous int32 vector aligned to the universe: [0..N-1] or a stable view."""
        ...

    def iter_ids(self) -> Iterator[int]:
        """Python iterator over ids (use ids_view() for vectorized paths)."""
        ids = self.ids_view()
        # Avoid Python range to respect any custom view; still zero-copy iteration.
        for i in range(ids.shape[0]):
            yield int(ids[i])

    # ---- ergonomic/compat (slow/materializing) ----

    def get_all_states(self) -> Any:
        """Lazy iterable over actual State objects. Fine for small N / debug; not for hot loops."""
        ...

    # ---- membership & conversions ----
    @abstractmethod
    def contains(self, s: S) -> bool:
        ...


    def get_id(self, s: S) -> int:
        ...


    def get_state_by_id(self, sid: int) -> S:
        ...

    # ---- current state ----
    @abstractmethod
    def get_state(self) -> S:
        ...

    @abstractmethod
    def set_state(self, state: S) -> None:
        ...

    # ---- metadata ----
    def dimension(self) -> int:
        """Optional dimensionality hint (vector states). Implement if meaningful."""
        return 1

    # ---- set algebra (optional to implement now) ----
    @abstractmethod
    def union_state_space(self, other: "StatSpace[S]") -> "StatSpace[S]":
        raise NotImplementedError

    @abstractmethod
    def intersection_state_space(self, other: "StatSpace[S]") -> "StatSpace[S]":
        raise NotImplementedError

    @classmethod
    def build_from_initial_state(self,cls,state: State):
        return cls(state,{state})

    @property
    def universe_id(self):
        """Stable identity for the state-id universe this space belongs to."""
        return getattr(self, "_universe_id", None)


    def remove_state(self,s:State):

        """Remove a state from this space."""
        pass



#------------Types of State Spaces -------------------#

class DiscreteSpace(StatSpace,ABC):  # tag/mixin
    @property
    def is_discrete(self) -> bool: return True

class ContinuousSpace(StatSpace,ABC):  # tag/mixin
    @property
    def is_discrete(self) -> bool: return False





# ---- A tiny zero-copy iterable view over stored states (no set/list creation) ----
@dataclass(frozen=True)
class _StatesView(Generic[S]):
    """Zero-copy iterable view over stored states (no set/list materialization)."""
    _id_to_state: Sequence[S]
    def __iter__(self) -> Iterator[S]:
        return iter(self._id_to_state)
    def __len__(self) -> int:
        return len(self._id_to_state)


def _infer_key(sample):
    if hasattr(sample, "value"):
        return lambda s: getattr(s, "value")
    if hasattr(sample, "coords"):
        return lambda s: tuple(getattr(s, "coords"))
    if hasattr(sample, "__len__") and hasattr(sample, "__getitem__"):
        return lambda s: tuple(s)
    return None  # try natural ordering, then fallback to repr

class DiscreteFiniteStatSpace(StatSpace[S]):
    """
    Finite, discrete state space with a bijection:
      state -> id (dict), id -> state (list)
    Deterministic id assignment:
      - If 'states' is a set/unsorted iterable, ordered by 'key' or repr(state).
      - If 'states' is a sequence, preserves given order.
    Fast path:
      - ids_view() -> np.int32 contiguous array for vectorized kernels.
    """

    def __init__(
        self,
        states: Iterable[S],
        current: Optional[S] = None,
        key: Optional[Callable[[S], Any]] = None,
        dim: Optional[int] = None,
    ) -> None:
        self._state_to_id: dict[S, int] = {}
        self._id_to_state: List[S] = []
        self._current_state: Optional[S] = None
        self._key = key
        self._dim = dim
        super().__init__(current,states)

    # --------- StatSpace: fast paths ----------
    def size(self) -> int:
        return len(self._id_to_state)

    def ids_view(self) -> np.ndarray:
        # Create on demand; zero extra storage.
        return np.arange(self.size(), dtype=np.int32)

    # --------- StatSpace: ergonomic/compat ----------
    def get_all_states(self) -> Iterable[S]:
        # Lazy, zero-copy iteration over actual states.
        return _StatesView(self._id_to_state)

    # --------- StatSpace: membership & conversions ----------
    def contains(self, s: S) -> bool:
        return s in self._state_to_id

    def get_id(self, s: S) -> int:
        return self._state_to_id[s]

    def get_state_by_id(self, sid: int) -> S:
        if sid < 0 or sid >= self.size():
            raise IndexError("State ID out of range.")
        return self._id_to_state[sid]

    # --------- StatSpace: current state ----------
    def get_state(self) -> S:
        assert self._current_state is not None
        return self._current_state

    def set_state(self, state: S) -> None:
        if state not in self._state_to_id:
            raise ValueError("State not in this space.")
        self._current_state = state

    # --------- StatSpace: metadata ----------
    def dimension(self) -> int:
        return int(self._dim or 1)

    # --------- Optional: set algebra (pure; returns new spaces) ----------
    def union_state_space(self, other: "DiscreteFiniteStatSpace[S]") -> "DiscreteFiniteStatSpace[S]":
        # Fast path: identical universe object → return a copy referencing same order
        if self._id_to_state is other._id_to_state:
            return DiscreteFiniteStatSpace(self._id_to_state, current=self._current_state, key=self._key, dim=self._dim)

        combined = list(self._id_to_state)
        seen = set(self._state_to_id)
        for s in other._id_to_state:
            if s not in seen:
                combined.append(s)
                seen.add(s)
        return DiscreteFiniteStatSpace(combined, current=self._current_state, key=self._key, dim=self._dim)

    def intersection_state_space(self, other: "DiscreteFiniteStatSpace[S]") -> "DiscreteFiniteStatSpace[S]":
        if self._id_to_state is other._id_to_state:
            return DiscreteFiniteStatSpace(self._id_to_state, current=self._current_state, key=self._key, dim=self._dim)

        other_set = set(other._state_to_id)
        inter = [s for s in self._id_to_state if s in other_set]
        if not inter:
            # Adjust policy if you want to allow empty spaces.
            raise ValueError("Intersection is empty.")
        return DiscreteFiniteStatSpace(inter, current=self._current_state, key=self._key, dim=self._dim)


    def build_from_states(self, states: Iterable[S], current: Optional[S] = None) -> "DiscreteFiniteStatSpace[S]":
        seq = list(states)
        if not seq:
            raise ValueError("DiscreteFiniteStatSpace cannot be empty.")

        # Sort deterministically if the input is not an ordered sequence
        if not isinstance(states, (list, tuple)):
            sort_key = self._key or _infer_key(seq[0])
            try:
                if sort_key is None:
                    seq.sort()
                else:
                    seq.sort(key=sort_key)
            except Exception:
                seq.sort(key=lambda x: repr(x))
            # cache inferred key for future rebuilds if not set
            if self._key is None and sort_key is not None:
                self._key = sort_key

        self._state_to_id.clear()
        self._id_to_state.clear()
        for s in seq:
            if s in self._state_to_id:
                continue
            self._state_to_id[s] = len(self._id_to_state)
            self._id_to_state.append(s)

        self._current_state = seq[0] if current is None or current not in self._state_to_id else current

        if self._dim is None:
            try:
                self._dim = int(len(self._id_to_state[0]))
            except Exception:
                self._dim = 1
        return self

    def remove_state(self, s: S) -> None:
        """
        Remove state `s` from the space.
        - If s not present → no-op
        - Re-index IDs to remain contiguous (0..N-1)
        - Adjust current_state safely
        """
        if s not in self._state_to_id:
            return  # nothing to do

        # 1) Get ID and remove from structures
        remove_id = self._state_to_id.pop(s)

        # 2) Remove from ID→state list
        self._id_to_state.pop(remove_id)

        # 3) Rebuild the mapping for IDs > removed_id
        #    Shift their IDs down by 1
        for st, sid in list(self._state_to_id.items()):
            if sid > remove_id:
                self._state_to_id[st] = sid - 1

        # 4) Fix current_state if needed
        if self._current_state == s:
            if self._id_to_state:
                # choose deterministic first state
                self._current_state = self._id_to_state[0]
            else:
                # No states left — degenerate space
                self._current_state = None

        # finalize (no need to recompute dim; states unchanged)


class DiscreteInfiniteStateSpace(DiscreteSpace, StatSpace[S], ABC):
    """Define membership by rule; optionally provide an iterator."""
    def __init__(self, initial_state: S, dim: int,
                 contains_fn: Callable[[S], bool],
                 iter_fn: Optional[Callable[[], Iterator[S]]] = None):
        super().__init__(initial_state)
        self._dim, self._contains, self._iter = dim, contains_fn, iter_fn

    def dimension(self) -> int: return self._dim
    def contains(self, s: S) -> bool: return self._contains(s)
    def get_all_states(self) -> Iterable[S]:
        if self._iter is None:
            return super().get_all_states()
        return self._iter()

class ContinuousFiniteStateSpace(ContinuousSpace, StatSpace[S], ABC):
    """Usually rare; if truly finite, you can still store a frozenset like above."""
    # implement similarly to DiscreteFiniteStateSpace if needed
    @classmethod
    def build_state_space(self, *args, **kwargs):
        '''Needs to be implemented by all subclasses to build the state space'''


# ------------------helper method to build space from reference ----------------#
Sin = TypeVar('Sin', bound=State)
Sout = TypeVar('Sout', bound=State)
def rebuild_like(
        proto_out_space: StatSpace[Sout],
        states: set[Sout],
        current: Optional[Sout] = None
) -> StatSpace[Sout]:
    """
        Rebuild a new space like 'proto' without using deepcopy.
        Prefer calling a finite-space constructor when possible.
    """
    # If you have a known finite implementation, use it here.
    # Example:
    # if isinstance(proto, DiscreteFiniteStateSpace):
    #     return DiscreteFiniteStateSpace(
    #         initial_state=current or next(iter(states), proto.get_state()),
    #         states=states
    #     )
    #
    # Generic fallback: rely on the proto API if it exposes build_from_states()
    new_space = type(proto_out_space)(proto_out_space.get_state())
    if hasattr(new_space, "build_from_states"):
            # type: ignore[attr-defined]
            new_space.build_from_states(set(states), current=current)  # finite contract
            return new_space
        # Last resort: return proto itself if truly no way to rebuild (documented limitation)
    return proto_out_space
