from abc import abstractmethod, ABC
from typing import Dict, Optional, Any, Generic, TypeVar

from fds import StatSpace, State
from fds.core.fds_state import IsAllowedState, Reachable
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem

S = TypeVar("S", bound=State)
class AffectedReachable(Reachable[S], ABC, Generic[S]):
    """
    Reachable that *knows the group members*. Keeps the standard get_reachable(state)
    entrypoint and delegates to the affected-specific implementation.
    """
    def __init__(
        self,
        members: Dict[str, FieldDynamicSystem],
        params: Optional[Dict[str, Any]] = None,
        is_allowed: Optional["IsAllowedState"] = None,
        from_allowed:bool=False
    ) -> None:
        super().__init__(params=params, is_allowed=is_allowed,from_allowed=from_allowed)
        self.members = members

    # standard API used by MultiStepField, etc.
    def get_reachable(self, state: S = None,within_state_space =False,from_allowed = False) -> StatSpace:
        return self.get_affected_reachable(state)

    @abstractmethod
    def get_affected_reachable(self, state: S) -> StatSpace:
        """Return the reachable *group* subspace from `state`, using self.members."""
        raise NotImplementedError
