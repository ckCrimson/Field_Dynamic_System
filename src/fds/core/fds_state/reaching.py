from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Generic, TypeVar, Set

from fds.core.fds_state.state import State
from fds.core.fds_state.state_space import rebuild_like, StatSpace

from fds.core.fds_state.reachable import Reachable

S = TypeVar("S", bound=State)


class Reaching(ABC, Generic[S]):
    """
    Computes the set of states that can reach a given `state` in one step.
    Can be implemented directly or derived from a `Reachable` instance.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        reachable: Optional["Reachable[S]"] = None,
    ):
        self.params: Dict[str, Any] = params or {}
        self.reachable: Optional["Reachable[S]"] = reachable

    @abstractmethod
    def get_reaching(self, state: S) -> StatSpace[S]:
        """
        Return a space of predecessors that can reach `state` in one step.
        Implement directly in subclasses when you have a natural 'inverse' rule.
        """
        raise NotImplementedError

    # ---------- Generic inverse via a Reachable (finite input scan) ----------

    def get_reaching_from_reachable(
        self,
        state: S,
        space: StatSpace[S],
        *,
        from_is_allowed: bool = False,
    ) -> StatSpace[S]:
        """
        Build predecessors by scanning all s in `space` and testing whether
        `state` is in reachable(s).

        Contract:
          - `space` must be finite (or at least expose get_all_states()).
          - We never mutate `space`; a NEW narrowed space is returned.
        """
        if self.reachable is None:
            # No provider: return an empty same-kind space (don’t mutate input)
            return rebuild_like(space, set(), current=None)

        # Ensure we can iterate all candidates
        try:
            candidates = space.get_all_states()
        except AttributeError as e:
            raise TypeError(
                "get_reaching_from_reachable requires a finite StateSpace with get_all_states()."
            ) from e

        predecessors: Set[S] = set()
        for s in candidates:
            r_space = (
                self.reachable.get_reachable_from_allowed(s, space)
                if from_is_allowed and hasattr(self.reachable, "get_reachable_from_allowed")
                else self.reachable.get_reachable(s)
            )
            # Use contains(): works for finite OR procedural reachable spaces
            if r_space.contains(state):
                predecessors.add(s)

        # Choose a stable current in the result
        current = space.get_state()
        if current not in predecessors and predecessors:
            current = next(iter(predecessors))
        elif not predecessors:
            current = None

        return rebuild_like(space, predecessors, current=current)
