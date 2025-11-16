from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

from fds.core.fds_state import StateSpaceMapping
from fds.core.fds_state.state_space import StatSpace, DiscreteFiniteStatSpace
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.core.fds_state.state import State


class AffectedSystemsMapping(StateSpaceMapping, ABC):
    """
    Bijective mapping between:
      - a CorrelatedState (collection of per-system states), and
      - a single affected/global state S_aff.

    get_affected_state  : CorrelatedState -> S_aff        (forward)
    get_system_state    : S_aff          -> CorrelatedState (inverse)
    """

    def __init__(self, *, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        super().__init__(params)

    # ----- forward (systems -> affected) -----
    @abstractmethod
    def get_affected_state(self, input_system_states: CorrelatedState) -> State:
        """Pack per-system states into one affected/global state."""
        raise NotImplementedError

    # ----- inverse (affected -> systems) -----
    @abstractmethod
    def get_system_state(self, input_affected_state: State) -> CorrelatedState:
        """Unpack an affected/global state back to per-system states."""
        raise NotImplementedError

    # ====== optimized forward-space mapping ======

    @staticmethod
    def _build_aff_space_like(
        proto: StatSpace,
        states: Set[State],
        current: Optional[State],
    ) -> StatSpace:
        """
        Create a NEW affected-state space of the same 'kind' as `proto` if possible,
        else fall back to DiscreteFiniteStatSpace.
        Never mutates `proto`.
        """
        # Prefer a subclass-provided build_like(states, current=...)
        bl = getattr(proto, "build_like", None)
        if callable(bl):
            return bl(states, current=current)

        # Generic finite fallback: discrete finite space of affected states
        return DiscreteFiniteStatSpace(states=states, current=current, key=None, dim=None)

    def forward_space(self, sys_space: StatSpace, aff_space_proto: StatSpace) -> StatSpace:
        """
        Map an input correlated-system state space -> affected-state space by applying the
        forward mapping pointwise. Preserves a sensible current element.

        Optimizations:
          - No deepcopy of `aff_space_proto`.
          - ID-first iteration when sys_space exposes ids_view()/get_state_by_id().
        """
        out_states: Set[State] = set()
        mapped_current: Optional[State] = None

        sys_current = sys_space.get_state()

        # Try id-based iteration first
        ids = getattr(sys_space, "ids_view", None)
        sid2state = getattr(sys_space, "get_state_by_id", None)

        if callable(ids) and callable(sid2state):
            # Fast path: iterate by ids
            for sid in ids():
                cs = sid2state(int(sid))    # CorrelatedState
                g = self.get_affected_state(cs)
                out_states.add(g)
                if mapped_current is None and cs == sys_current:
                    mapped_current = g
        else:
            # Fallback: generic state iteration
            for cs in sys_space.get_all_states():
                g = self.get_affected_state(cs)
                out_states.add(g)
                if mapped_current is None and cs == sys_current:
                    mapped_current = g

        if not out_states:
            # No mapped states → return an empty-ish clone of the prototype type
            # (caller should define what an "empty" affected space means)
            return self._build_aff_space_like(aff_space_proto, set(), current=None)

        # Choose a stable current if we didn't hit the sys_current
        if mapped_current is None:
            try:
                mapped_current = min(out_states)
            except Exception:
                mapped_current = next(iter(out_states))

        # Build a fresh affected-state space of the same "kind" as the prototype
        return self._build_aff_space_like(aff_space_proto, out_states, current=mapped_current)

    # Quick sanity checks you can call in tests
    def round_trip_ok_forward(self, cs: CorrelatedState) -> bool:
        """Check inverse(forward(cs)) == cs."""
        try:
            return self.get_system_state(self.get_affected_state(cs)) == cs
        except Exception:
            return False
