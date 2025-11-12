from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Optional, Any

from fds import State, StatSpace
from fds.affecting.affecting_systems_framework.affected_reachable import AffectedReachable
from fds.core.fds_state import Reaching
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem

S = TypeVar("S", bound=State)
class AffectedReaching(Reaching[S], ABC, Generic[S]):
    """
    Reaching helper that also knows the group members. Keeps parent constructor
    and adds a clear affected-specific entrypoint.
    """
    def __init__(
        self,
        members: Dict[str, FieldDynamicSystem],
        params: Optional[Dict[str, Any]] = None,
        reachable: Optional[AffectedReachable[S]] = None,
    ) -> None:
        super().__init__(params=params, reachable=reachable)
        self.members = members

    @abstractmethod
    def get_affected_reaching(self) -> StatSpace:
        """Return the *group* reaching state space (e.g., within L steps)."""
        raise NotImplementedError
