import copy
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from fds import State, StatSpace
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.core.fds_state import StateSpaceMapping


class AffectedSystemsMapping(StateSpaceMapping,ABC):
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
    def get_system_state(self, input_affected_state: State) -> List[Dict[str,State]]:
        """Unpack an affected/global state back to per-system states."""
        raise NotImplementedError

    # ====== convenience utilities ======

    def forward_space(self, sys_space: StatSpace, aff_space_proto: StatSpace) -> StatSpace:
        """
        Map an input correlated-system state space -> affected-state space by applying the
        forward mapping pointwise. Preserves a sensible current element.
        """
        out = copy.deepcopy(aff_space_proto)
        out_states = set()
        mapped_current = None

        sys_current = sys_space.get_state()
        for cs in sys_space.get_all_states():
            g = self.get_affected_state(cs)
            out_states.add(g)
            if mapped_current is None and cs == sys_current:
                mapped_current = g

        if not out_states:
            return out  # empty clone

        if mapped_current is None:
            mapped_current = next(iter(out_states))
        out.build_from_states(out_states, current=mapped_current)
        return out

    # Quick sanity checks you can call in tests
    def round_trip_ok_forward(self, cs: CorrelatedState) -> bool:
        """Check inverse(forward(cs)) == cs."""
        try:
            return self.get_system_state(self.get_affected_state(cs)) == cs
        except Exception:
            return False

